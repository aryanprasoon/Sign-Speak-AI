import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def preprocess_data(df):
    """
    Normalizes the coordinates to make them scale and translation invariant.
    """
    X = []
    y = df['label'].values
    
    # We expect 21 sets of x,y,z -> 63 features
    feature_cols = [col for col in df.columns if col != 'label']
    raw_coords = df[feature_cols].values
    
    for row in raw_coords:
        # Reshape to (21, 3)
        coords = row.reshape(21, 3)
        
        # 1. Translation: center at wrist (landmark 0)
        wrist = coords[0].copy()
        translated = coords - wrist
        
        # 2. Scaling: normalize by max distance from wrist
        max_dist = np.max(np.linalg.norm(translated, axis=1))
        if max_dist > 0:
            normalized = translated / max_dist
        else:
            normalized = translated
            
        # Flatten back to 63 features
        X.append(normalized.flatten())
        
    return np.array(X), y

def train_model():
    dataset_path = os.path.join(SCRIPT_DIR, "dataset.csv")
    print(f"Loading dataset from '{dataset_path}'...")
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    df = pd.read_csv(dataset_path)
    
    print("Preprocessing data...")
    X, y = preprocess_data(df)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    model_path = os.path.join(SCRIPT_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved successfully as '{model_path}'")

if __name__ == "__main__":
    train_model()
