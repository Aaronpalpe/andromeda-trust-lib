from sklearn.ensemble import RandomForestClassifier
import pickle

#los datos train y test están en train.csv y test.csv, los cargamos

import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
X_train = train.drop(columns=["Target"])
y_train = train["Target"]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(clf, f)
