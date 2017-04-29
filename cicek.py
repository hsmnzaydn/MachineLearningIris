import pandas as pd
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









