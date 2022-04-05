import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd

df = pd.read_csv(r'D:\DATASCIENCE\DeepLearning-DS\Chapter 03\01. MLP Binary Classification\data.csv')

# print(df.head())

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 16)  # input = num of features, out up to you
        self.hidden1 = nn.Linear(16, 8) # input = out of previus layer, output up to you
        self.hidden2 = nn.Linear(8, 4) # input = out of previus layer, output up to you
        self.output = nn.Linear(4, 1) # input = out of previus layer, it depends on the task
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): #forwardpass
        first_layer = self.input_layer(x)
        act1 = self.sigmoid(first_layer)
        second_layer = self.hidden1(act1)
        act2 = self.sigmoid(second_layer)
        third_layer = self.hidden2(act2)
        act3 = self.sigmoid(third_layer)
        out_layer = self.output(act3)
        # prediction = self.sigmoid(out_layer)
        return self.sigmoid(out_layer)


loss = nn.BCELoss()
model = Classifier()
optimizer = T.optim.Adam(model.parameters(), lr=0.001)
x = np.random.rand(1000,2)
y = np.random.randint(0, 2, 1000)
x_tensor = T.tensor(x).float()
y_true_tensor = T.tensor(y).float()
y_true_tensor = y_true_tensor.view(1000,1) # view function is the same as reshape in numpy
y_pred_tensor = model(x_tensor)
loss_value = loss(y_pred_tensor, y_true_tensor)
print(f"Initial loss: {loss_value.item():.2f}")