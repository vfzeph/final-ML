import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from fit_models import prepare_data

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'D:\\car-main4\\car-main3-main\\car-main3\\car-main\\static'

import matplotlib
matplotlib.use('Agg')

# Directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "model_outputs")
os.makedirs(models_dir, exist_ok=True)
scaler_path = os.path.join(base_dir, "static", "scaler.pkl")  

# Load scaler and models
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
models = {f.split('.')[0]: joblib.load(os.path.join(models_dir, f))
          for f in os.listdir(models_dir) if f.endswith('.pkl')}

def preprocess_data(data):
    features, _, _ = prepare_data(data)
    return features

def make_predictions(processed_data):
    predictions = {model_name: model.predict(processed_data)
                   for model_name, model in models.items()}
    probabilities = {model_name: model.predict_proba(processed_data)[:, 1]
                     for model_name, model in models.items() if hasattr(model, 'predict_proba')}
    return predictions, probabilities

def generate_confusion_matrix_plot(y_true, y_pred, model_name):
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')

        plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{model_name}_confusion_matrix.png")
        plt.savefig(plot_path, format='png', bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error in generating/saving confusion matrix for {model_name}: {e}")
        return None

def generate_roc_curve_plot(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{model_name}_roc_curve.png")
    plt.savefig(plot_path, format='png', bbox_inches='tight')
    plt.close()

    return plot_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            try:
                data = pd.read_csv(file)
                predictions, conf_matrix_plots, roc_curve_plots = process_data_and_predict(data)
                return render_template('results.html', predictions=predictions, conf_matrix_plots=conf_matrix_plots, roc_curve_plots=roc_curve_plots)
            except Exception as e:
                flash(f"An error occurred: {e}")
                return redirect(url_for('index'))
        else:
            flash("Invalid file. Please upload a CSV file.")
            return redirect(url_for('index'))

    return redirect(url_for('index'))

def process_data_and_predict(data):
    X, y, _ = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    predictions, probabilities = make_predictions(X_test)
    conf_matrix_plots = {model_name: generate_confusion_matrix_plot(y_test, model_pred, model_name)
                         for model_name, model_pred in predictions.items()}
    roc_curve_plots = {model_name: generate_roc_curve_plot(y_test, y_scores, model_name)
                       for model_name, y_scores in probabilities.items()}
    return predictions, conf_matrix_plots, roc_curve_plots

if __name__ == '__main__':
    app.run(port=5001, debug=True)
