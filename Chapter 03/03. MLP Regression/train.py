#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

from model import Regression

model = Regression(8)

x_train, x_test, y_train, y_test = dh.load_data(r'Chapter 03\03. MLP Regression\data\turkish_stocks.csv')

epochs = 100


criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

loss_train = []
loss_test = []


for epoch in range(epochs):

    running_train_loss = 0
    running_test_loss = 0

    x_trbh, x_tsbh, y_trbh, y_tsbh = dh.to_batches(x_train, x_test, y_train, y_test, 20)
    for i in range(x_trbh.shape[0]):
        

        optimizer.zero_grad()
        pred=model.forward(x_trbh[i])
        train_loss=criterion(pred,y_trbh[i])
        train_loss.backward()
        optimizer.step()


        model.eval()
        with torch.no_grad():

            test_pred = model.forward(x_tsbh)

            test_loss = criterion(test_pred, y_tsbh)


        model.train()

        running_train_loss += train_loss.item()
        running_test_loss += test_loss.item()

        #loss_train.append(train_loss.item())
        #loss_test.append(test_loss.item())
       
        print(f'Epoch: {epoch + 1} | loss: {train_loss.item()} | test loss: {test_loss.item()}' )

    loss_train.append(running_train_loss/len(x_trbh))
    loss_test.append(running_test_loss/len(x_tsbh))

plt.plot(loss_train, label='train Loss')
plt.plot(loss_test, label='test Loss')

plt.legend()
plt.show()







# Remember to validate your model: with torch.no_grad() ...... model.eval .........model.train
