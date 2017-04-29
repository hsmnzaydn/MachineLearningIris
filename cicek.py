import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from sklearn import metrics
import sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

dosya_yolu=input("Dosya yolu nedir?")
data_verileri = dosya_yolu

veriler = pd.read_csv(data_verileri, header = 0)
hedef=veriler[veriler.columns[4:5]].values


original_headers = list(veriler.columns.values)
veriler = veriler._get_numeric_data()
veriler_to_matrix = veriler.as_matrix()
kaynak=veriler_to_matrix

model=DecisionTreeClassifier()
model.fit(kaynak, hedef)

sepallength=input("SepalLenght?")
sepalwidth=input("SepalWidth?")
petallength=input("PetalLenght?")
petalwidth=input("PetalWidth?")


print(model.predict([sepallength,sepalwidth,petallength,petalwidth]))









