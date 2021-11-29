import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

random_value = 33

test = pd.read_csv('./data/pulsar_data_test.csv')
train = pd.read_csv('./data/pulsar_data_train.csv')

pre_train = train.dropna()
y = pre_train['target_class']
# print(y.head())
X = pre_train.drop(['target_class'], axis=1)
# print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.3, random_state=random_value)

clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
true_positives = cm[0, 0]
false_positives = cm[0, 1]
print('True positives', true_positives)
print('False positives', false_positives)

y_pred_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:,1])

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Predicting a Pulsar Star classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
