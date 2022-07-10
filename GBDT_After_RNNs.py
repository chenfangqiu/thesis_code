import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt 

Xdata=pd.read_csv('data.csv',usecols=[2,3,4,5])
Ydata=pd.read_csv('data.csv',usecols=[1])

X=Xdata.values
Y=Ydata.values.ravel()

gbr = GradientBoostingRegressor(n_estimators=10000, max_depth=3, min_samples_split=4, learning_rate=0.01)
gbr.fit(X,Y)
y_gbr = gbr.predict(X)
acc = gbr.score(X,Y)

dot_data = export_graphviz(gbr.estimators_[0, 0], out_file=None, filled=False, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.view(filename="1")