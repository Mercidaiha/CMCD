# coding: utf-8

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
from EduCDM import CDM
from .loss import PairSCELoss, HarmonicLoss


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point, user_id_pair=None):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        if user_id_pair is not None:
            stu_emb_pair = self.student_emb(user_id_pair)
            stat_emb_pair = torch.sigmoid(stu_emb_pair)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))
        if user_id_pair is not None:
            return output_1.view(-1), stat_emb, stat_emb_pair
        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, zeta=0.5):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.zeta = zeta

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        score_loss_function = nn.BCELoss()
        theta_loss_function = PairSCELoss()
        loss_function = HarmonicLoss(self.zeta)
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
                batch_count += 1
                # user_id, item_id, knowledge_emb, y = batch_data
                user_id, item_id, knowledge_emb, y, user_id_pair, pos = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                user_id_pair: torch.Tensor = user_id_pair.to(device)
                pos: torch.Tensor = pos.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                
                uid_pairs = torch.chunk(user_id_pair, user_id_pair.size(1), dim=1)
                poses = torch.chunk(pos, pos.size(1), dim=1)

                loss_theta = 0
                for i in range(user_id_pair.size(1)):
                    predicted_response, predicted_theta, predicted_theta_pair = self.ncdm_net(user_id, item_id, knowledge_emb, torch.squeeze(uid_pairs[i], dim=-1))
                    # loss = loss_function(pred, y)
                    predicted_response = torch.Tensor(predicted_response)
                    predicted_theta = torch.Tensor(predicted_theta)
                    predicted_theta_pair = torch.Tensor(predicted_theta_pair)
                    # loss_score = score_loss_function(predicted_response, y)
                    # print("2\n")
                    # print(predicted_theta_pair.shape)
                    # print(pos.shape)
                    # print(response)
                    # print(pos)
                    loss1, count1 = theta_loss_function(predicted_theta, predicted_theta_pair, poses[i])
                    loss_theta += loss1
                    count += count1
                loss_score = score_loss_function(predicted_response, y)
                loss = loss_function(loss_score, loss_theta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            print("[Epoch %d] average loss: %.6f, Count: %d" % (epoch_i, float(np.mean(epoch_losses)), count))

            if test_data is not None:
                rmse, mae, auc, accuracy = self.eval(test_data, device=device)
                print("[Epoch %d] rmse: %.6f, mae: %.6f, auc: %.6f, accuracy: %.6f" % (epoch_i, rmse, mae, auc, accuracy))

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        loss_function = nn.BCELoss()
        losses = []
        
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y: torch.Tensor = y.to(device)
            loss = loss_function(pred, y)
            losses.append(loss.mean().item())
            
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        print("[Valid Loss] %.6f" % (float(np.mean(losses))))
        return np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
