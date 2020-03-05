import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.linear_model import LinearRegression

height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
weight=[42,44,49,55,53,58,60,64,66,69]

def train(x,y):
    model=LinearRegression()
    model.fit(height,weight)
    m = model.coef_[0]
    b = model.intercept_
    return model

model = train(height,weight)
print(model.coef_[0],model.intercept_)


def pred(height):
        
    weight=model.predict(height)
    return weight
  
print(model.predict([[5.5]]))
