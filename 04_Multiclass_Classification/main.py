import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Value %i' % label)
try:
    plt.show()
except:
    print('fail')
else:
    print('not fail')

sample = digits.images[0]
target = digits.target[0]
# print(sample, target)

# 1 reshape images
print('----1----')
images = digits.images.reshape(len(digits.images), 64)

# 2 create train and test sets
print('----2----')
X_train, X_test, y_train, y_test = train_test_split(images, digits.target, test_size=0.3, random_state=1)

# 3 logistic classifier
print('----3----')
clf = SGDClassifier(loss="log", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

# 4 confusion matrix
print('----4----')
clf_report = classification_report(y_test, prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(clf_report)

# 5 compare test and train
print('----5----')
train_prediction = clf.predict(X_train)
clf_train_report = classification_report(y_train, train_prediction, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(clf_train_report)

# 6 play
print('----6----')


def classify_and_report(X_train, y_train, X_test, y_test):
    classifier = SGDClassifier(loss="log", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    return report


X1_train, X1_test, y1_train, y1_test = train_test_split(images, digits.target, test_size=0.2, random_state=10)
print(classify_and_report(X1_train, y1_train, X1_test, y1_test))

X2_train, X2_test, y2_train, y2_test = train_test_split(images, digits.target, test_size=0.3, random_state=3)
print(classify_and_report(X2_train, y2_train, X2_test, y2_test))

X3_train, X3_test, y3_train, y3_test = train_test_split(images, digits.target, test_size=0.35, random_state=2)
print(classify_and_report(X3_train, y3_train, X3_test, y3_test))
