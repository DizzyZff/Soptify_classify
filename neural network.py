import pandas as pd
import sqlite3
import plotly.express as px
import torch
from sklearn import metrics

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

X_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
y_train = y_train.astype('int')
X_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']
y_test = y_test.astype('int')


class nn(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(nn, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


input_size = 5
hidden_size = 100
output_size = 10

params = {
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size
}

model = nn(**params)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train = torch.from_numpy(X_train.values).float()
y_train = torch.from_numpy(y_train.values).long()
X_test = torch.from_numpy(X_test.values).float()
y_test = torch.from_numpy(y_test.values).long()

epochs = 2000

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch: ', epoch, 'Loss: ', loss.item())

with torch.no_grad():
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))

# # Test the model

model.eval()
