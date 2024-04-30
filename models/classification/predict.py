import sys

import joblib
import numpy as np
import pandas as pd

CSV_DIR = '../../out/csv'
CHECKPOINT_DIR = 'checkpoint'


def load_models():
    models = [
        "Logistic_Regression_model.pkl",
        "Random_Forest_model.pkl",
        "SVM_model.pkl",
        "Gradient_Boosting_model.pkl",
        "Decision_Tree_model.pkl",
        "K-Nearest_Neighbors_model.pkl"
    ]
    loaded_models = {model: joblib.load(f"{CHECKPOINT_DIR}/{model}") for model in models}
    return loaded_models


def make_prediction(trimmed_data, models):
    predictions = np.array([model.predict(trimmed_data) for model in models.values()])

    # Majority voting
    final_prediction = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
    return final_prediction


def trim_data(input_data_path):
    input_data = pd.read_csv(input_data_path)
    trimmed_data = input_data.drop('name', axis=1)
    trimmed_data['respectsISNT'] = trimmed_data['respectsISNT'].astype(int)
    return trimmed_data


def main():
    if len(sys.argv) != 2:
        print('Usage: python predict.py <input_data_csv>')
        return

    input_data_csv = sys.argv[1]
    input_data_path = f"{CSV_DIR}/{input_data_csv}"
    trimmed_data = trim_data(input_data_path)

    models = load_models()
    prediction = make_prediction(trimmed_data, models)
    diagnosis = "Glaucoma" if prediction else "Healthy"
    print(f"Predicted diagnosis: {diagnosis}")


if __name__ == "__main__":
    main()
