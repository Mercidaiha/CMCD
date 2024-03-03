# coding: utf-8
# 2021/4/23 @ tongshiwei

import logging
import numpy as np
import torch
from EduCDM import CDM
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from ..irt import irt3pl
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from .loss import PairSCELoss, HarmonicLoss


class IRTNet(nn.Module):
    def __init__(self, user_num, item_num, value_range, a_range, irf_kwargs=None):
        super(IRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, 1)
        self.a = nn.Embedding(self.item_num, 1)
        self.b = nn.Embedding(self.item_num, 1)
        self.c = nn.Embedding(self.item_num, 1)
        self.value_range = value_range
        self.a_range = a_range

    def forward(self, user, item, user_pair=None):
        # print(user)
        # print(user_pair)
        theta = torch.squeeze(self.theta(user), dim=-1)
        if user_pair is not None:
            theta_pair = torch.squeeze(self.theta(user_pair), dim=-1)
        # print(theta.shape)
        a = torch.squeeze(self.a(item), dim=-1)
        b = torch.squeeze(self.b(item), dim=-1)
        c = torch.squeeze(self.c(item), dim=-1)
        c = torch.sigmoid(c)
        if self.value_range is not None:
            theta = self.value_range * (torch.sigmoid(theta) - 0.5)
            theta_pair = self.value_range * (torch.sigmoid(theta_pair) - 0.5)
            b = self.value_range * (torch.sigmoid(b) - 0.5)
        if self.a_range is not None:
            a = self.a_range * torch.sigmoid(a)
        else:
            a = F.softplus(a)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The value_range or a_range is too large.')
        if user_pair is not None:
            return self.irf(theta, a, b, c, **self.irf_kwargs), torch.sigmoid(theta), torch.sigmoid(theta_pair)
        else:
            return self.irf(theta, a, b, c, **self.irf_kwargs)

    @classmethod
    def irf(cls, theta, a, b, c, **kwargs):
        return irt3pl(theta, a, b, c, F=torch, **kwargs)


class GIRT(CDM):
    def __init__(self, user_num, item_num, value_range=None, a_range=None, zeta=0.5):
        super(GIRT, self).__init__()
        self.irt_net = IRTNet(user_num, item_num, value_range, a_range)
        self.zeta = zeta

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001) -> ...:
        self.irt_net = self.irt_net.to(device)
        score_loss_function = nn.BCELoss(reduction="none")
        theta_loss_function = PairSCELoss(reduction="none")
        loss_function = HarmonicLoss(self.zeta)

        trainer = torch.optim.Adam(self.irt_net.parameters(), lr)

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response, user_id_pair, pos, fake, group = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                user_id_pair: torch.Tensor = user_id_pair.to(device)
                pos: torch.Tensor = pos.to(device)
                fake: torch.Tensor = fake.to(device)
                group: torch.Tensor = group.to(device)
                response: torch.Tensor = response.to(device)
                
                predicted_response, predicted_theta, predicted_theta_pair = self.irt_net(user_id, item_id, user_id_pair)
                predicted_response = torch.Tensor(predicted_response)
                predicted_theta = torch.Tensor(predicted_theta)
                predicted_theta_pair = torch.Tensor(predicted_theta_pair)
                
                loss_score = score_loss_function(predicted_response, response)
                
                
                # 根据每个样本的条件来计算损失函数
                loss = torch.zeros_like(loss_score)  # 初始化损失为全零张量
                # 对每个样本的条件进行逐元素判断
                mask_1 = (fake == 0) & (group == 0)
                mask_2 = (fake == 1) & (group == 0)
                mask_3 = (fake == 0) & (group == 1)
                # 根据条件计算损失
                if mask_1.any():
                    loss_theta_1 = theta_loss_function(predicted_theta[mask_1], predicted_theta_pair[mask_1], pos[mask_1])
                    loss[mask_1] = loss_function(loss_score[mask_1], loss_theta_1)
                if mask_2.any():
                    loss_theta_2 = theta_loss_function(predicted_theta[mask_2], predicted_theta_pair[mask_2], pos[mask_2])
                    loss[mask_2] = loss_function(loss_score[mask_2], loss_theta_2)
                if mask_3.any():
                    loss[mask_3] = loss_function(loss_score[mask_3], torch.zeros(loss_score[mask_3].shape[0], device=loss_score[mask_3].device))

                # 只对非零损失进行反向传播
                non_zero_loss_indices = (loss != 0)
                if non_zero_loss_indices.any():
                    # back propagation
                    trainer.zero_grad()
                    (loss[non_zero_loss_indices]).mean().backward()
                    trainer.step()

                    losses.append(loss[non_zero_loss_indices].mean().item())
                    
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None:
                rmse, mae, auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] rmse: %.6f, mae: %.6f, auc: %.6f, accuracy: %.6f" % (e, rmse, mae, auc, accuracy))

    def eval(self, test_data, device="cpu") -> tuple:
        self.irt_net = self.irt_net.to(device)
        self.irt_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.irt_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.irt_net.train()
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.irt_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.irt_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
