import sys

import joblib
import numpy as np
import pandas as pd
import shap

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


def shap_explain(model_name, model, trimmed_data):
    X = trimmed_data.astype(float)

    # Initialize SHAP explainer based on the model type
    if "Tree" in model_name or "Random_Forest" in model_name or "Gradient_Boosting" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="dot")

    elif "Logistic_Regression" in model_name:
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer(X)
        shap.plots.waterfall(shap_values[0])

    elif "SVM" in model_name or "K-Nearest_Neighbors" in model_name:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar")

    else:
        raise NotImplementedError(f"SHAP explainer not implemented for {model_name}")


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

    # explain with shap
    for model_name, model in models.items():
        shap_explain(model_name, model, trimmed_data)


if __name__ == "__main__":
    main()
