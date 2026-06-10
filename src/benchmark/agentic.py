import asyncio
import random
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, read_time_series_full, get_db_url
from src.agent.client import run as run_agent
from src.benchmark.run_baselines import calculate_metrics

async def main():
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
    
    for table in ts_tables:
        print(f"Agentic benchmarking table: {table}")
        df = read_time_series_full(table)
        if 'label' not in df.columns:
            print(f"Skipping {table}: no 'label' column")
            continue
            
        y_true = df['label'].fillna(0).astype(int).values
        n = len(y_true)
        
        try:
            report = await run_agent(table)
            
            y_pred = np.zeros(n, dtype=int)
            y_score = np.zeros(n, dtype=float)
            
            for anom in report.anomalies:
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
                    "method": "agentic",
                    "p": p,
                    "r": r,
                    "f1": f1,
                    "roc": roc,
                    "pr": pr
                })
        except Exception as e:
            print(f"Failed on {table}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
