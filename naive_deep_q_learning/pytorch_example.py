import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, 128) # *input_dims unpacks the tuple
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        # stochastic gradient descent with momentum
        # self.parameters() comes from nn.Module
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda_is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        # Pytorch assumes data to be tensors
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2) # no activation function here, crossentropyloss does it

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad()
        # the lower case tensor is preserving the data type of the source data
        data = T.tensor(data).to(self.device) 
        labels = T.tensor(labels).to(self.device)

        predicitons = self.forward(data)

        cost = self.loss(predicitons, labels)

        cost.backward()
        self.optimizer.step()
