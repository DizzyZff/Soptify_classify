import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.svm import SVC

data_path = 'musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

x_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
y_train = y_train.astype('int')
x_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']
y_test = y_test.astype('int')

# svc
svm = SVC(decision_function_shape='ovr')
svm.fit(x_train, y_train)
svm_pred_labels = svm.predict(x_test)

# metrics
svm_acc = metrics.accuracy_score(y_test, svm_pred_labels)
svm_f1 = metrics.f1_score(y_test, svm_pred_labels, average='weighted')
svm_precision = metrics.precision_score(y_test, svm_pred_labels, average='weighted')
svm_recall = metrics.recall_score(y_test, svm_pred_labels, average='weighted')
print("SVM Accuracy: ", svm_acc)
print("SVM F1: ", svm_f1)
print("SVM Precision: ", svm_precision)
print("SVM Recall: ", svm_recall)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(10):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, svm_pred_labels)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

fig, ax = plt.subplots()
for i in range(10):
    ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic example')
ax.legend(loc="lower right")
plt.show()




