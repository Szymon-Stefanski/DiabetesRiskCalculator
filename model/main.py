import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")


def random_forest(data):
    X = data.drop(["Diabetes_binary"], axis=1)
    y = data["Diabetes_binary"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    return 0

def clean_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates()
    data = data.astype(int)
    return data


def main():
    data = clean_data()
    model = random_forest(data)


if __name__ == '__main__':
    main()
