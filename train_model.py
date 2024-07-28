import joblib

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


data = load_iris()
x,y = data.data,data.target
print(data)
model = RandomForestClassifier()
model.fit(x,y)


joblib.dump(model,'model.joblib')