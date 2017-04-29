import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import tree
import pydotplus
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.tree import DecisionTreeClassifier


input_file = "iris.csv"

df = pd.read_csv(input_file, header = 0)
print()
target=df[df.columns[4:5]].values


original_headers = list(df.columns.values)
df = df._get_numeric_data()
numpy_array = df.as_matrix()
source=numpy_array

model=DecisionTreeClassifier()
model.fit(source,target)


print(model.predict([5.1,3.7,1.5,0.5]))









