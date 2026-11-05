"""
Autonomous Execution File (Monsoon 2026 PR Course Submission)
Executes entire pipeline end-to-end and outputs test metrics.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

def run_project_pipeline():
    print("Executing Autonomous Pattern Recognition Pipeline...")
    col_path = "Faults27x7_var" if os.path.exists("Faults27x7_var") else "/kaggle/input/datasets/thanhvv/dataset-steel-plates-faults/Faults27x7_var"
    data_path = "Faults.NNA" if os.path.exists("Faults.NNA") else "/kaggle/input/datasets/thanhvv/dataset-steel-plates-faults/Faults.NNA"
    
    with open(col_path, "r", encoding="utf-8") as f:
        col_names = f.read().splitlines()
    df = pd.read_csv(data_path, sep=r"\s+", header=None, names=col_names)
    
    targets = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    valid = df[targets].sum(axis=1) > 0
    df = df[valid].reset_index(drop=True)
    
    X = df.drop(columns=targets).copy()
    y = df[targets].values.argmax(axis=1)
    
    # Feature extraction
    x_range = (X["X_Maximum"] - X["X_Minimum"]).clip(lower=1.0)
    y_range = (X["Y_Maximum"] - X["Y_Minimum"]).clip(lower=1.0)
    X["Aspect_Ratio"] = y_range / x_range
    total_perim = (X["X_Perimeter"] + X["Y_Perimeter"]).clip(lower=1.0)
    X["Shape_Compactness"] = (total_perim ** 2) / (X["Pixels_Areas"] + 1e-5)
    X["Thickness_Perimeter_Ratio"] = total_perim / X["Steel_Plate_Thickness"].clip(lower=1.0)
    X["Luminosity_Spread"] = X["Maximum_of_Luminosity"] - X["Minimum_of_Luminosity"]
    
    sel = [
        'Length_of_Conveyer', 'TypeOfSteel_A400', 'TypeOfSteel_A300',
        'Steel_Plate_Thickness', 'Log_X_Index', 'Edges_Y_Index', 'LogOfAreas',
        'Sum_of_Luminosity', 'X_Minimum', 'Orientation_Index',
        'Outside_X_Index', 'Pixels_Areas', 'Square_Index', 'X_Maximum',
        'X_Perimeter', 'Minimum_of_Luminosity', 'Empty_Index', 'Y_Maximum',
        'Aspect_Ratio', 'Shape_Compactness', 'Thickness_Perimeter_Ratio', 'Luminosity_Spread'
    ]
    X_sub = X[[c for c in sel if c in X.columns]]
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_sub, y):
        X_tr, X_va = X_sub.iloc[train_idx], X_sub.iloc[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        clf = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(n_estimators=220, max_depth=5, learning_rate=0.06, eval_metric="mlogloss", random_state=42, verbosity=0)),
                ('lgb', LGBMClassifier(n_estimators=220, max_depth=6, num_leaves=31, learning_rate=0.06, random_state=42, verbose=-1)),
                ('cat', CatBoostClassifier(iterations=250, depth=6, learning_rate=0.06, random_seed=42, verbose=0)),
                ('et', ExtraTreesClassifier(n_estimators=250, max_depth=14, random_state=42, n_jobs=-1))
            ],
            voting='soft', weights=[2.5, 2.5, 2.0, 1.0]
        )
        clf.fit(X_tr, y_tr)
        preds = np.asarray(clf.predict(X_va)).ravel().astype(int)
        scores.append(accuracy_score(y_va, preds) * 100)
        
    print(f"SUCCESS: 10-Fold CV Accuracy = {np.mean(scores):.2f}% ± {np.std(scores):.2f}%")

if __name__ == '__main__':
    run_project_pipeline()
