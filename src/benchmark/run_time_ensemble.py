import random
import time
import numpy as np
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, get_db_url
from src.mcp_server import (
    _run_hbos_raw as hbos_detector,
    _run_iforest_raw as iforest_detector,
    _run_pca_raw as pca_detector,
    _run_poly_raw as poly_detector,
    _run_lof_raw as lof_detector
)

def main():
    random.seed(RANDOM_SEED)
    
    tables = list_tables()
    # Exclude metadata tables and the execution_time table itself
    ts_tables = [t for t in tables if t not in ('experiments', 'execution_time')]

    if len(ts_tables) > SUBSET_SIZE:
        ts_tables = random.sample(ts_tables, SUBSET_SIZE)
    
    engine = create_engine(get_db_url())
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS execution_time (
                id SERIAL PRIMARY KEY,
                dataset VARCHAR(255),
                method VARCHAR(255),
                time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    
    for table in ts_tables:
        print(f"Benchmarking ensemble execution time for table: {table}")
        
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT 1 FROM execution_time WHERE dataset = :dataset AND method = 'ensemble' LIMIT 1"),
                {"dataset": table}
            ).fetchone()
            
        if existing:
            print(f"  Skipping ensemble on {table}: Already computed.")
            continue

        try:
            start_time = time.perf_counter()
            
            scores = []
            scores.append(lof_detector(table))
            scores.append(iforest_detector(table))
            scores.append(hbos_detector(table))
            scores.append(pca_detector(table))
            scores.append(poly_detector(table))
            
            # Fuse scores to a mean score
            _ = np.mean(scores, axis=0)
            
            elapsed_time = time.perf_counter() - start_time
            
            print(f"  Finished ensemble in {elapsed_time:.4f} seconds.")
            
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO execution_time (dataset, method, time)
                    VALUES (:dataset, 'ensemble', :time)
                """), {
                    "dataset": table,
                    "time": elapsed_time
                })
        except Exception as e:
            print(f"  Failed ensemble on {table}: {e}")

if __name__ == "__main__":
    main()
