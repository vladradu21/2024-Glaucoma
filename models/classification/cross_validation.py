import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

# Hyperparameters etc.
CSV_DIR = '../../data/csv'


def handle_data(data_path):
    data = pd.read_csv(data_path)

    data.drop('name', axis=1, inplace=True)  # Remove 'name' column
    data['respectsISNT'] = data['respectsISNT'].astype(int)  # Convert 1/0
    data['hasGlaucoma'] = data['hasGlaucoma'].astype(int)  # Convert 1/0

    x = data.drop('hasGlaucoma', axis=1)
    y = data['hasGlaucoma']

    return x, y


def evaluate_models(x, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    kf = KFold(n_splits=5, shuffle=True)

    results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
        results[name] = cv_scores
        print(f"{name}: Average Accuracy = {cv_scores.mean() * 100:.2f}%")

    return results


def main():
    data_path = f"{CSV_DIR}/metrics.csv"
    x, y = handle_data(data_path)
    evaluate_models(x, y)


if __name__ == "__main__":
    main()
