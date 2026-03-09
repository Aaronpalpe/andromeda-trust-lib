from sklearn.svm import SVC
import pickle
import pandas as pd

# cargar datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_train = train.drop(columns=["Target"])
y_train = train["Target"]

# modelo SVM
clf = SVC(
    kernel="rbf",
    probability=True,   # necesario si luego usas predict_proba
    random_state=42
)

clf.fit(X_train, y_train)

# guardar modelo
with open("model_SVM.pkl", "wb") as f:
    pickle.dump(clf, f)