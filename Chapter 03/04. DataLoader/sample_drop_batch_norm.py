import torch as T
import torch.nn as nn
import torch.nn.functional as F

class TestNet(nn.Module):

    def __init__(self):
        super(TestNet, self).__init__()

        self.fc1 = nn.Linear(128, 512)
        self.bn1 = nn.BatchNorm1d(512) # Batcn Norm for the first layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256) # Batcn Norm for the secon layer
        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)  # Batcn Norm for the third layer
        self.out = nn.Linear(64, 10)
        self.do  = nn.Dropout(0.2, inplace=True)

    def forward(self, x):
        x = F.relu(self.bn1(self.do(self.fc1(x))))
        x = F.relu(self.bn2(self.do(self.fc2(x))))
        x = F.relu(self.bn3(self.do(self.fc3(x))))
        return self.out(x)

# test it out

x = T.rand((100, 128))

model = TestNet()

logits = model.forward(x)

probs = F.softmax(logits, dim=1)

print(probs.topk(1, dim=1))

