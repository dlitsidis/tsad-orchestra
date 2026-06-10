import random
import numpy as np
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, read_time_series_full, get_db_url
from src.mcp_server import lof_detector, hbos_detector, iforest_detector, pca_detector, poly_detector
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

def calculate_metrics(y_true, y_score, sliding_window):
    metrics = get_metrics(score=y_score, labels=y_true, slidingWindow=sliding_window)
    # Convert numpy types to float for db insertion
    return {k: float(v) for k, v in metrics.items()}

def main():
    random.seed(RANDOM_SEED)
    
    tables = list_tables()
    ts_tables = [t for t in tables if t != 'experiments']

    if len(ts_tables) > SUBSET_SIZE:
        ts_tables = random.sample(ts_tables, SUBSET_SIZE)

    ts_tables = ['545_smap_id_15_sensor_tr_1173_1st_2750']      
    
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

        data = df.iloc[:,1].astype(float)
        
        slidingWindow = find_length(data)

        y_true = df['label'].fillna(0).astype(int).values
        n = len(y_true)
        
        for name, func in detectors.items():
            print(f"  Running {name}...")
            try:
                y_score = func(table, _return_raw=True)
                
                metrics = calculate_metrics(y_true, y_score, slidingWindow)
                
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO experiments (
                            dataset_name, method, auc_roc, auc_pr, precision, recall, f, 
                            precision_at_k, rprecision, rrecall, rf, r_auc_roc, r_auc_pr, 
                            vus_roc, vus_pr, affiliation_precision, affiliation_recall
                        )
                        VALUES (
                            :dataset, :method, :auc_roc, :auc_pr, :precision, :recall, :f, 
                            :precision_at_k, :rprecision, :rrecall, :rf, :r_auc_roc, :r_auc_pr, 
                            :vus_roc, :vus_pr, :affiliation_precision, :affiliation_recall
                        )
                    """), {
                        "dataset": table,
                        "method": name,
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
                print(f"    Failed {name} on {table}: {e}")

if __name__ == "__main__":
    main()
