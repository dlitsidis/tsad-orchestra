import random
import numpy as np
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, read_time_series_full, get_db_url
from src.mcp_server import (
    _run_hbos_raw as hbos_detector,
    _run_iforest_raw as iforest_detector,
    _run_pca_raw as pca_detector,
    _run_poly_raw as poly_detector,
    _run_lof_raw as lof_detector
)
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

def calculate_metrics(y_true, y_score, sliding_window):
    metrics = get_metrics(score=y_score, labels=y_true, slidingWindow=sliding_window)
    # Convert to float
    return {k: float(v) for k, v in metrics.items()}

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
                auc_roc FLOAT,
                auc_pr FLOAT,
                precision FLOAT,
                recall FLOAT,
                f FLOAT,
                precision_at_k FLOAT,
                rprecision FLOAT,
                rrecall FLOAT,
                rf FLOAT,
                r_auc_roc FLOAT,
                r_auc_pr FLOAT,
                vus_roc FLOAT,
                vus_pr FLOAT,
                affiliation_precision FLOAT,
                affiliation_recall FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    
    for table in ts_tables:
        print(f"Benchmarking table: {table} with ensemble")
        
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT 1 FROM experiments WHERE dataset_name = :dataset AND method = 'ensemble' LIMIT 1"),
                {"dataset": table}
            ).fetchone()
            
        if existing:
            print(f"  Skipping ensemble on {table}: Already computed.")
            continue

        df = read_time_series_full(table)
        data = df.iloc[:,1].astype(float)
        
        slidingWindow = find_length(data)
        y_true = df['label'].fillna(0).astype(int).values
        n = len(y_true)
        
        try:
            scores = []
            
            # Individual scores
            print("    Running lof...")
            scores.append(lof_detector(table))
            
            print("    Running iforest...")
            scores.append(iforest_detector(table))
            
            print("    Running hbos...")
            scores.append(hbos_detector(table))
            
            print("    Running pca...")
            scores.append(pca_detector(table))
            
            print("    Running poly...")
            scores.append(poly_detector(table))
            
            # Fuse scores
            y_score = np.mean(scores, axis=0)
            
            metrics = calculate_metrics(y_true, y_score, slidingWindow)
            
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO experiments (
                        dataset_name, method, auc_roc, auc_pr, precision, recall, f, 
                        precision_at_k, rprecision, rrecall, rf, r_auc_roc, r_auc_pr, 
                        vus_roc, vus_pr, affiliation_precision, affiliation_recall
                    )
                    VALUES (
                        :dataset, 'ensemble', :auc_roc, :auc_pr, :precision, :recall, :f, 
                        :precision_at_k, :rprecision, :rrecall, :rf, :r_auc_roc, :r_auc_pr, 
                        :vus_roc, :vus_pr, :affiliation_precision, :affiliation_recall
                    )
                """), {
                    "dataset": table,
                    "auc_roc": metrics.get("AUC_ROC"),
                    "auc_pr": metrics.get("AUC_PR"),
                    "precision": metrics.get("Precision"),
                    "recall": metrics.get("Recall"),
                    "f": metrics.get("F"),
                    "precision_at_k": metrics.get("Precision_at_k"),
                    "rprecision": metrics.get("Rprecision"),
                    "rrecall": metrics.get("Rrecall"),
                    "rf": metrics.get("RF"),
                    "r_auc_roc": metrics.get("R_AUC_ROC"),
                    "r_auc_pr": metrics.get("R_AUC_PR"),
                    "vus_roc": metrics.get("VUS_ROC"),
                    "vus_pr": metrics.get("VUS_PR"),
                    "affiliation_precision": metrics.get("Affiliation_Precision"),
                    "affiliation_recall": metrics.get("Affiliation_Recall")
                })
        except Exception as e:
            print(f"  Failed ensemble on {table}: {e}")

if __name__ == "__main__":
    main()
