import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import torch
import sqlite3
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#load train and test datasets
data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

print(train.dtypes)

#Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(train_x)
    loss = criterion(output, train_y)
    loss.backward()
    optimizer.step()
    if (epoch+1)%10 == 0:
        print("Epoch: ", epoch+1, "Loss: ", loss.item())

#test
y_pred = model(test_x)
y_pred = y_pred.detach().numpy()
y_pred = np.argmax(y_pred, axis=1)
print("Accuracy: ", sklearn.metrics.accuracy_score(test_y, y_pred))




