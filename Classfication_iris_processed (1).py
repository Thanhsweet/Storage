import pandas as pd
#from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


data = pd.read_csv('C:/Users/Admin/source/VSCode/Artifical_Intelligence_Ki3/K_NN/iris.txt')
# profile = ProfileReport(data, title="Iris Processed Report")
# profile.to_file("iris_processed_report.html")

# result1 = data.describe()
# result2 = data.info()
# print(result2)

target = "Species"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

#Train model
model = KNeighborsClassifier(n_neighbors=3) # thuat toan KNN
model.fit(x_train, y_train) # huan luyen mo hinh
y_pred = model.predict(x_test)  # du doan

# print(y_pred)
# print(len(y_pred))
# print("------------")
# print(y_test.values)

for i, j in zip(y_pred, y_test):
    print("Predicted : {}. Actual value: {}".format(i, j))

print(classification_report(y_test, y_pred))