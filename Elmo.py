import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch

class Elmo(nn.Module):

    #compute the elmo representation

    def __init__(self, max_len, num_layers, trainable = True):
        super(Elmo,self).__init__()

        self.weights = Parameter(torch.randn(num_layers+1, max_len),requires_grad=trainable)
        self.gamma = Parameter(torch.randn(max_len, 1), requires_grad=trainable)

    def forward(self, state_list):

        normal_weights = F.softmax(self.weights,dim=1).permute(1,0).unsqueeze(1)
        all_state = torch.cat([state.unsqueeze(2) for state in state_list], dim=2)
        weighted_state = torch.matmul(normal_weights,all_state).squeeze(2)
        elmo_repre = self.gamma * weighted_state

        return elmo_repre