# model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        SVC: Trained model
    """
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train):
    """
    Train a Multilayer Perceptron model.
    Args:
        X_train (pd.DataFrame): Training data features
        y_train (pd.Series): Training data labels
    Returns:
        MLPClassifier: Trained model
    """
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    """
    Save the trained model to a file.
    Args:
        model: Trained model
        model_path (str): Path where the model should be saved
    """
    joblib.dump(model, model_path)

def load_model(model_path):
    """
    Load a trained model from a file.
    Args:
        model_path (str): Path to the saved model
    Returns:
        Model: Loaded model
    """
    return joblib.load(model_path)
