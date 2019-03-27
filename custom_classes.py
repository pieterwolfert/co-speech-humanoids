import torch
import torch.nn as nn
from torch.autograd import Variable

class CustomLoss(torch.nn.Module):
    def __init__(self, alpha, beta):
        super(CustomLoss,self).__init__()
        self.loss = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self,pose_t, pose_t_1,y):
        """Implements the loss function for Yoon et al. 2018
        where the loss the combined loss of mean squared error,
        continuity (based on the previous pose) and
        the negative variance of the current pose.
        """
        pose_t = pose_t.view(-1)
        pose_t_1 = pose_t_1.view(-1)
        y = y.view(-1)
        loss_mse = self.loss(pose_t, y)
        loss_continuity = Variable(torch.sum(torch.abs(pose_t - pose_t_1)) / 19)
        loss_variance = Variable(torch.neg(torch.var(pose_t)))
        total_loss = loss_mse + self.alpha * loss_continuity + self.beta *\
            loss_variance
        return total_loss
