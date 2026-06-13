import argparse
import asyncio
import random
from sqlalchemy import create_engine, text

from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
from src.utils.db import list_tables, get_db_url
from src.agent.client import run as run_agent

async def run_tool_usage_benchmark(
    skip_validator: bool,
    subset_size: int,
    method_name: str,
    seed: int
):
    random.seed(seed)
    
    # Get tables
    tables = list_tables()
    
    if len(tables) > subset_size:
        tables = random.sample(tables, subset_size)
        
    print(f"Starting tool usage benchmark on {len(tables)} tables...")
    print(f"Pipeline: {'Agentic (No Validator)' if skip_validator else 'Agentic (With Validator)'}")
    print(f"Target DB Method/Model name: '{method_name}'\n")

    engine = create_engine(get_db_url())
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tool_usage (
                id SERIAL PRIMARY KEY,
                dataset_name VARCHAR(255),
                method VARCHAR(255),
                tool_name VARCHAR(255),
                count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

    total_tool_counts = {}

    for i, table in enumerate(tables, start=1):
        print(f"[{i}/{len(tables)}] Running agent on table: {table}...")
        try:
            report = await run_agent(table, skip_validator=skip_validator)
            
            tool_counts = getattr(report, "tool_counts", {})
            if tool_counts:
                # Format string
                counts_str = ", ".join(f"{t}: {c}" for t, c in sorted(tool_counts.items()))
                print(f"  -> Tool usage: {counts_str}")
                
                # Write to database
                with engine.begin() as conn:
                    # Clear existing runs
                    conn.execute(
                        text("DELETE FROM tool_usage WHERE dataset_name = :dataset AND method = :method"),
                        {"dataset": table, "method": method_name}
                    )
                    # Insert tool counts
                    for tool_name, count in tool_counts.items():
                        conn.execute(text("""
                            INSERT INTO tool_usage (dataset_name, method, tool_name, count)
                            VALUES (:dataset, :method, :tool_name, :count)
                        """), {
                            "dataset": table,
                            "method": method_name,
                            "tool_name": tool_name,
                            "count": count
                        })
                        total_tool_counts[tool_name] = total_tool_counts.get(tool_name, 0) + count
            else:
                print("  -> No tools called during this run.")
        except Exception as e:
            print(f"  -> Failed on {table}: {e}")

    print("\n" + "=" * 40)
    print("           TOOL USAGE SUMMARY           ")
    print("=" * 40)
    if total_tool_counts:
        # Print table
        print(f"{'Tool Name':<30} | {'Total Invocations':<15}")
        print("-" * 50)
        for tool_name, count in sorted(total_tool_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{tool_name:<30} | {count:<15}")
    else:
        print("No tool invocations were recorded.")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description="Measure TSAD agent tool usage and save to DB.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agentic", action="store_true", help="Measure tool usage with validator")
    group.add_argument("--agentic-no-validator", action="store_true", help="Measure tool usage without validator")
    
    parser.add_argument("--subset-size", type=int, default=SUBSET_SIZE, help=f"Number of tables to run (default: {SUBSET_SIZE})")
    parser.add_argument("--method-name", type=str, default=None, help="Custom method name to save in DB")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help=f"Random seed (default: {RANDOM_SEED})")

    args = parser.parse_args()

    skip_validator = args.agentic_no_validator
    
    if args.method_name:
        method_name = args.method_name
    else:
        method_name = "no_validator_tool_usage" if skip_validator else "agentic_tool_usage"

    asyncio.run(run_tool_usage_benchmark(
        skip_validator=skip_validator,
        subset_size=args.subset_size,
        method_name=method_name,
        seed=args.random_seed
    ))

if __name__ == "__main__":
    main()
