import asyncio
import random
import time
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, get_db_url
from src.agent.client import run as run_agent

async def main():
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
    
    for table in ts_tables:
        print(f"Benchmarking agentic execution time for table: {table}")
        
        with engine.connect() as conn:
            existing = conn.execute(
                text("SELECT 1 FROM execution_time WHERE dataset = :dataset AND method = 'new_new_agentic' LIMIT 1"),
                {"dataset": table}
            ).fetchone()
            
        if existing:
            print(f"  Skipping agentic on {table}: Already computed.")
            continue

        try:
            start_time = time.perf_counter()
            _ = await run_agent(table)
            elapsed_time = time.perf_counter() - start_time
            
            print(f"  Finished agentic in {elapsed_time:.4f} seconds.")
            
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO execution_time (dataset, method, time)
                    VALUES (:dataset, 'new_new_agentic', :time)
                """), {
                    "dataset": table,
                    "time": elapsed_time
                })
        except Exception as e:
            print(f"  Failed agentic on {table}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
