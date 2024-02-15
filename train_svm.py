from sklearn import svm
from sklearn import datasets
import pickle


model = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
model.fit(X, y)

# save the model to disk
filename = 'models/model.sav'
pickle.dump(model, open(filename, 'wb'))

for i in range(100):
  print(X[i], y[i])