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

# recurrent neural network
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.rnn = torch.nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


input_size = 12
hidden_size = round(50000/((12*10)*5))
num_classes = 10

model = NeuralNet(input_size, hidden_size, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train = torch.from_numpy(train.drop(['music_genre'], axis=1).values).float()
y_train = torch.from_numpy(train['music_genre'].values).long()
X_test = torch.from_numpy(test.drop(['music_genre'], axis=1).values).float()
y_test = torch.from_numpy(test['music_genre'].values).long()
train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

accuracy_list = []
for epoch in range(5000):
    inputs = X_train.to(device)
    labels = y_train.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, 5000, loss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print('Accuracy of the model on the test data: {} %'.format(accuracy))

