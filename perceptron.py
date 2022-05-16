#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
from flask import Flask, request
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])
df = df.loc[df.target!=2,['sepal length (cm)', 'petal length (cm)', 'target']]

class Perceptron:
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update=self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

siec = Perceptron()
a=df.iloc[:,:2].values
b=df.target.values
siec.fit(a,b)

with open("model_rta.pkl", "wb") as jc:
    pickle.dump(siec, jc)
app = Flask(__name__)

@app.route("/api/predict", methods = ['GET'])
def prediction():

    with open("model_rta.pkl", "rb") as jc:    
        model = pickle.load(jc)

    sl = float(request.args.get("sl"))
    pl = float(request.args.get("pl"))
    
    zbior=[sl, pl]
    x = int(model.predict(zbior))
    
    if x == 1:
        nazwa =  "versicolor"
    else:
        nazwa = "setosa"
    return nazwa

app.run(port='5000', host="0.0.0.0")
#http://0.0.0.0:5000/api/predict?&sl=4.5&pl=3.2           


