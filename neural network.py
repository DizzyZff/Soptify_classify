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
# ,'mode_Major','mode_Minor'

# multiclass classification neural network
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 12
hidden_size = 100
num_classes = 10

model = NeuralNet(input_size, hidden_size, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train = torch.from_numpy(X_train.values).float()
y_train = torch.from_numpy(y_train.values).long()
X_test = torch.from_numpy(X_test.values).float()
y_test = torch.from_numpy(y_test.values).long()
train = torch.utils.data.TensorDataset(X_train, y_train)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

accuracy_list = []
for epoch in range(5000):
    inputs = X_train.to(device)
    labels = y_train.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, 1000, loss.item()))

    if (epoch + 1) % 10 == 0:
        # acc on genre 1
        correct = 0
        total = 0
        with torch.no_grad():
            inputs = X_test.to(device)
            labels = y_test.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print('Accuracy of the network on the test images: {} %'.format(accuracy))

# plot accuracy
fig = px.line(x=range(0,5000,10), y=accuracy_list)
fig.show()

# confusion matrix
with torch.no_grad():
    inputs = X_test.to(device)
    labels = y_test.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(metrics.confusion_matrix(labels.cpu(), predicted.cpu()))

# save model
torch.save(model.state_dict(), 'model.ckpt')








