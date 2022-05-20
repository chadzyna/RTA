#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from flask import Flask, request
from uczenie import Perceptron

app = Flask(__name__)

@app.route("/api/predict", methods = ['GET'])
def prediction():
    
    with open("model.pkl", "rb") as jc:    
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


    
#app.run(port='5004')
if __name__ == '__main__':
    app.run(port='5000', host="0.0.0.0")
    #app.run(port='5000')
#http://0.0.0.0:5000/api/predict?&sl=4.5&pl=3.2 
#http://127.0.0.1:5000/api/predict?&sl=4.5&pl=3.2 
#http://127.0.0.1:5004/api/predict?&sl=4.5&pl=3.2 
# In[ ]:




