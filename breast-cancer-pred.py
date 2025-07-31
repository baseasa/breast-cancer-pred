import pandas as pd 
import seaborn as sns

data = pd.read_csv(r"C:\spyder\datasets\breast cancer pred\data.csv")
data.head()
data.info()
data.describe()

sns.heatmap(data.isnull())

data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.head()

data.diagnosis = [1 if value == "M" else 0 for value in data.diagnosis]
data["diagnosis"] = data['diagnosis'].astype("category", copy=False)
data["diagnosis"].value_counts().plot(kind="bar")

# divide into target variable and predictors
y = data["diagnosis"] #our target variable
X = data.drop(["diagnosis"], axis=1)

#NORMALIZE THE DATA 
from sklearn.preprocessing import StandardScaler

#create a scaler obj
scaler = StandardScaler()

#fit the scaler to the data and  transform the data
X_scaled = scaler.fit_transform(X)

#SPLIT THE DATA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

#TRAIN THE MODEL
from sklearn.linear_model import LogisticRegression

#create the lr model
lr = LogisticRegression()

#train the model on the training data
lr.fit(X_train, y_train)

#predict the target variable on test data
y_pred = lr.predict(X_test)

#EVALUATION OF THE MODEL
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))































