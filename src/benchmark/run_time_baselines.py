import random
import time
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
    # Exclude metadata tables
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
    
    detectors = {
        'lof': lof_detector,
        'hbos': hbos_detector,
        'iforest': iforest_detector,
        'pca': pca_detector,
        'poly': poly_detector
    }
    
    for table in ts_tables:
        print(f"Benchmarking execution time for table: {table}")
        
        for name, func in detectors.items():
            print(f"  Running {name}...")
            
            with engine.connect() as conn:
                existing = conn.execute(
                    text("SELECT 1 FROM execution_time WHERE dataset = :dataset AND method = :method LIMIT 1"),
                    {"dataset": table, "method": name}
                ).fetchone()
                
            if existing:
                print(f"    Skipping {name} on {table}: Already computed.")
                continue

            try:
                start_time = time.perf_counter()
                _ = func(table)
                elapsed_time = time.perf_counter() - start_time
                
                print(f"    Finished {name} in {elapsed_time:.4f} seconds.")
                
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO execution_time (dataset, method, time)
                        VALUES (:dataset, :method, :time)
                    """), {
                        "dataset": table,
                        "method": name,
                        "time": elapsed_time
                    })
            except Exception as e:
                print(f"    Failed {name} on {table}: {e}")

if __name__ == "__main__":
    main()
