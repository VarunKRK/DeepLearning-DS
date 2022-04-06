import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

# YOUR CODE HERE

class Regression(nn.Module):
    def __init__(self, input):
        super(Regression, self).__init__()
        self.input_layer = nn.Linear(input, 56)  # input = num of features, out up to you
        self.hidden1 = nn.Linear(56, 49) # input = out of previus layer, output up to you
        self.hidden2 = nn.Linear(49, 8) # input = out of previus layer, output up to you
        self.output = nn.Linear(8, 1) # input = out of previus layer, it depends on the task
        
    
    def forward(self, x):
        first_layer = self.input_layer(x)
        act1 = F.relu(first_layer)
        second_layer = self.hidden1(act1)
        act2 = F.relu(second_layer)
        third_layer = self.hidden2(act2)
        act3 = F.relu(third_layer)
        out_layer = self.output(act3)
        #x = F.relu(out_layer)
        return out_layer





        