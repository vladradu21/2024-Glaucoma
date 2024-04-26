import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC

from cross_validation import handle_data

# Hyperparameters etc.
CSV_DIR = '../../data/csv'
CHECKPOINT_DIR = 'checkpoint'


def train_and_save_models(x, y, n_splits=5):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, model in models.items():
        accuracies = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            accuracies.append(accuracy)

        average_accuracy = np.mean(accuracies)
        print(f"{name} Average Accuracy: {average_accuracy * 100:.2f}%")

        # Save each model
        model_path = f"{CHECKPOINT_DIR}/{name.replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_path)


def main():
    data_path = f"{CSV_DIR}/metrics.csv"
    x, y = handle_data(data_path)
    train_and_save_models(x, y)


if __name__ == "__main__":
    main()
