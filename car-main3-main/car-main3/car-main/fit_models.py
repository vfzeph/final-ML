import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, \
    precision_score, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def create_directories():
    output_dir = "model_outputs"
    roc_auc_dir = os.path.join(output_dir, "model_roc_auc")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(roc_auc_dir).mkdir(parents=True, exist_ok=True)
    return output_dir, roc_auc_dir

def load_data():
    data_path = str(Path(__file__).parents[0] / "data/car_eval_dataset.csv")
    df = pd.read_csv(data_path)
    return df

def prepare_data(df):
    data = df.drop(columns=['Unnamed: 0'])
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == "object":
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le
    X = data.drop("class", axis=1)
    y = data["class"]
    return X, y, label_encoders

def save_label_encoders(label_encoders, directory):
    for column, encoder in label_encoders.items():
        encoder_filename = os.path.join(directory, f'{column}_encoder.pkl')
        joblib.dump(encoder, encoder_filename)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def initialize_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, penalty='l2', solver='lbfgs',
                                                  multi_class='ovr'),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=1),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2,
                                                min_samples_leaf=1),
        "SVM": SVC(probability=True, C=1.0, kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='uniform')
    }
    return models

def train_models(X_train, y_train, models):
    param_grids = {
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ['l2'],
            "solver": ['lbfgs', 'saga'],
        },
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        clf = GridSearchCV(model, param_grids.get(name, {}), cv=5, scoring='accuracy')
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_
        trained_models[name] = best_model

    return trained_models


def save_models(trained_models, output_dir):
    for name, model in trained_models.items():
        # Save the trained model
        model_filename = os.path.join(output_dir, f'{name.replace(" ", "_").lower()}_model.pkl')
        joblib.dump(model, model_filename)

def generate_confusion_matrices(X_test, y_test, models):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print(f"{name} Confusion Matrix:\n{cm}")

def calculate_metrics(y_true, y_pred_prob, y_pred):
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1], multi_class='ovr')
    else:
        n_classes = len(np.unique(y_true))
        roc_auc = 0.0
        specificity = 0.0

        for i in range(n_classes):
            y_true_class = (y_true == i)
            y_pred_proba_class = y_pred_prob[:, i]
            roc_auc += roc_auc_score(y_true_class, y_pred_proba_class, average='weighted') / n_classes
            
            tn, fp, fn, tp = confusion_matrix(y_true_class, (y_pred == i)).ravel()
            specificity += tn / (tn + fp) / n_classes

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)  # Set zero_division to 1
    return roc_auc, accuracy, recall, precision, specificity


def calculate_sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return sensitivity

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    return specificity

def calculate_fpr(y_true, y_pred_prob, num_classes):
    fpr = {}
    for i in range(num_classes):
        fpr[i], _ = roc_curve((y_true == i), y_pred_prob[:, i])
    return fpr

def calculate_fpr_tpr(y_true, y_pred_prob, num_classes):
    fpr = {}
    tpr = {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i), y_pred_prob[:, i])
    return fpr, tpr


def evaluate_models(X_test, y_test, trained_models):
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        roc_auc, accuracy, recall, precision, specificity = calculate_metrics(y_test, y_pred_proba, y_pred)
        fpr, tpr = calculate_fpr_tpr(y_test, y_pred_proba, len(np.unique(y_test)))  # Calculate FPR and TPR

        results[name] = {
            "ROC AUC": roc_auc,
            "Accuracy": accuracy,
            "Sensitivity (Recall)": recall,
            "Precision": precision,
            "Specificity": specificity,
            "FPR": fpr,
            "TPR": tpr  # Add TPR to the results
        }

    return results



def print_model_comparison(results):
    print("\nModel Comparison:")
    print("{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format("Model", "ROC AUC", "Accuracy", "Recall", "Precision", "Sensitivity"))
    for model, metrics in results.items():
        print("{:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.2f}".format(
            model, metrics["ROC AUC"], metrics["Accuracy"], metrics["Sensitivity (Recall)"], metrics["Precision"], metrics["Specificity"]))


def main():
    output_dir, roc_auc_dir = create_directories()
    df = load_data()
    X, y, label_encoders = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_test = scale_data(X_train, X_test)
    models = initialize_models()

    # Train models
    trained_models = train_models(X_train, y_train, models)

    # Save models
    save_models(trained_models, output_dir)

    label_encoders_dir = os.path.join(output_dir, "label_encoders")
    Path(label_encoders_dir).mkdir(parents=True, exist_ok=True)
    save_label_encoders(label_encoders, label_encoders_dir)

    # Generate confusion matrices
    generate_confusion_matrices(X_test, y_test, trained_models)

    # Evaluate models
    results = evaluate_models(X_test, y_test, trained_models)

    # Print model comparison
    print_model_comparison(results)

    # Plot ROC curves
    plot_roc_curves(results)

def plot_roc_curves(results):
    plt.figure(figsize=(10, 8))
    for name, metrics in results.items():
        fpr = metrics["FPR"]  # Use FPR from the results
        tpr = metrics["TPR"]
        for i in range(len(fpr)):
            plt.plot(fpr[i], tpr[i], label=f'{name} Class {i}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

    file_path = os.path.join('static', 'roc_curve.png')

    plt.savefig('car-main3/car-main/static/roc_curve.png')

if __name__ == '__main__':
    main()