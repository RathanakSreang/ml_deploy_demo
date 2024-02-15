import pickle

filename = 'models/model.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
X_test = [6.8, 2.8, 4.8, 1.4] #1
# [4.8 3.4 1.9 0.2] 0
result = loaded_model.predict([X_test])
print(result)