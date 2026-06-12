import argparse
import asyncio

from src.benchmark.agentic import main as run_agentic_main
from src.benchmark.agentic_no_validator import main as run_agentic_no_validator_main
from src.benchmark.ensemble import main as run_ensemble_main
from src.benchmark.run_baselines import main as run_baselines_main
from src.benchmark.run_time_agentic import main as run_time_agentic_main
from src.benchmark.run_time_agentic_no_validator import main as run_time_agentic_no_validator_main
from src.benchmark.run_time_baselines import main as run_time_baselines_main
from src.benchmark.run_time_ensemble import main as run_time_ensemble_main


async def run_token_usage_agentic_benchmark(skip_validator: bool, method_name: str):
    import random

    from src.agent.client import run as run_agent
    from src.benchmark.configurations import RANDOM_SEED, SUBSET_SIZE
    from src.utils.db import list_tables

    random.seed(RANDOM_SEED)
    tables = list_tables()
    ts_tables = [t for t in tables if t != "experiments"]

    if len(ts_tables) > SUBSET_SIZE:
        ts_tables = random.sample(ts_tables, SUBSET_SIZE)

    print(f"Starting agentic token usage benchmark on {len(ts_tables)} datasets...")

    for i, table in enumerate(ts_tables, start=1):
        print(f"[{i}/{len(ts_tables)}] Running agent on table: {table}...")
        try:
            report = await run_agent(table, skip_validator=skip_validator, db_method_name=method_name)
            print(f"  -> Tokens used: {getattr(report, 'total_tokens', 'N/A')}")
        except Exception as e:
            print(f"  -> Failed on {table}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline for TSAD Orchestra")
    parser.add_argument("--baselines", action="store_true", help="Run the baseline algorithms (without agent)")
    parser.add_argument("--ensemble", action="store_true", help="Run the ensemble baseline")
    parser.add_argument("--agentic", action="store_true", help="Run the agentic solution")
    parser.add_argument(
        "--agentic-no-validator", action="store_true", help="Run the agentic solution without the validator (ablation study)"
    )
    parser.add_argument("--time-baselines", action="store_true", help="Measure baseline execution times")
    parser.add_argument("--time-ensemble", action="store_true", help="Measure ensemble execution times")
    parser.add_argument("--time-agentic", action="store_true", help="Measure agentic execution times")
    parser.add_argument(
        "--time-agentic-no-validator", action="store_true", help="Measure agentic without validator execution times"
    )
    parser.add_argument("--token-usage", action="store_true", help="Measure and record agent token usage")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run baselines, ensemble, agentic, agentic-no-validator, and measure all execution times",
    )

    args = parser.parse_args()

    # Create a list of flags to check
    flags = [
        args.baselines,
        args.ensemble,
        args.agentic,
        args.agentic_no_validator,
        args.time_baselines,
        args.time_ensemble,
        args.time_agentic,
        args.time_agentic_no_validator,
        args.token_usage,
        args.all,
    ]

    if not any(flags):
        print(
            "Please specify at least one flag: --baselines, --ensemble, --agentic, --agentic-no-validator, --time-baselines, --time-ensemble, --time-agentic, --time-agentic-no-validator, --token-usage, or --all"
        )
        parser.print_help()
        return

    if args.token_usage:
        if args.agentic:
            print("\n=== Running Agentic Token Usage Benchmark ===")
            asyncio.run(run_token_usage_agentic_benchmark(skip_validator=False, method_name="agentic"))
        elif args.agentic_no_validator:
            print("\n=== Running Agentic (No Validator) Token Usage Benchmark ===")
            asyncio.run(run_token_usage_agentic_benchmark(skip_validator=True, method_name="agentic_no_validator"))
        else:
            print("Please specify --agentic or --agentic-no-validator with --token-usage.")
        return

    if args.baselines or args.all:
        print("\n=== Running Baselines ===")
        run_baselines_main()

    if args.ensemble or args.all:
        print("\n=== Running Ensemble Baseline ===")
        run_ensemble_main()

    if args.agentic or args.all:
        print("\n=== Running Agentic Solution ===")
        asyncio.run(run_agentic_main())

    if args.agentic_no_validator or args.all:
        print("\n=== Running Agentic Solution (No Validator — Ablation) ===")
        asyncio.run(run_agentic_no_validator_main())

    if args.time_baselines or args.all:
        print("\n=== Measuring Baseline Execution Times ===")
        run_time_baselines_main()

    if args.time_ensemble or args.all:
        print("\n=== Measuring Ensemble Execution Times ===")
        run_time_ensemble_main()

    if args.time_agentic or args.all:
        print("\n=== Measuring Agentic Execution Times ===")
        asyncio.run(run_time_agentic_main())

    if args.time_agentic_no_validator or args.all:
        print("\n=== Measuring Agentic (No Validator) Execution Times ===")
        asyncio.run(run_time_agentic_no_validator_main())


if __name__ == "__main__":
    main()
