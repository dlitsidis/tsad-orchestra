import os
import sys
import random
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, read_time_series_full, get_db_url
from src.mcp_server import lof_detector, hbos_detector, iforest_detector, pca_detector, poly_detector

def calculate_metrics(y_true, y_pred, y_score=None):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc_roc = 0.0
    auc_pr = 0.0
    
    if len(set(y_true)) > 1:
        score_to_use = y_score if y_score is not None else y_pred
        auc_roc = roc_auc_score(y_true, score_to_use)
        auc_pr = average_precision_score(y_true, score_to_use)
        
    return float(precision), float(recall), float(f1), float(auc_roc), float(auc_pr)

def main():
    random.seed(RANDOM_SEED)
    
    tables = list_tables()
    ts_tables = [t for t in tables if t != 'experiments']
    
    if len(ts_tables) > SUBSET_SIZE:
        ts_tables = random.sample(ts_tables, SUBSET_SIZE)
        
    engine = create_engine(get_db_url())
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS experiments (
                id SERIAL PRIMARY KEY,
                dataset_name VARCHAR(255),
                method VARCHAR(255),
                precision FLOAT,
                recall FLOAT,
                f1_score FLOAT,
                auc_roc FLOAT,
                auc_pr FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    
    detectors = {
        'lof': lof_detector,
        'hbos': hbos_detector,
        'iforest': iforest_detector,
        'pca': pca_detector,
        'poly': poly_detector
    }
    
    for table in ts_tables:
        print(f"Benchmarking table: {table}")
        df = read_time_series_full(table)
        if 'label' not in df.columns:
            print(f"Skipping {table}: no 'label' column")
            continue
            
        y_true = df['label'].fillna(0).astype(int).values
        n = len(y_true)
        
        for name, func in detectors.items():
            print(f"  Running {name}...")
            try:
                result = func(table)
                
                y_pred = np.zeros(n, dtype=int)
                y_score = np.zeros(n, dtype=float)
                
                for anom in result.anomalies:
                    if 0 <= anom.index < n:
                        y_pred[anom.index] = 1
                        y_score[anom.index] = anom.score
                        
                p, r, f1, roc, pr = calculate_metrics(y_true, y_pred, y_score)
                
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO experiments (dataset_name, method, precision, recall, f1_score, auc_roc, auc_pr)
                        VALUES (:dataset, :method, :p, :r, :f1, :roc, :pr)
                    """), {
                        "dataset": table,
                        "method": name,
                        "p": p,
                        "r": r,
                        "f1": f1,
                        "roc": roc,
                        "pr": pr
                    })
            except Exception as e:
                print(f"    Failed {name} on {table}: {e}")

if __name__ == "__main__":
    main()
