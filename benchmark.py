import argparse
import asyncio

from src.benchmark.run_baselines import main as run_baselines_main
from src.benchmark.agentic import main as run_agentic_main
from src.benchmark.agentic_no_validator import main as run_agentic_no_validator_main
from src.benchmark.ensemble import main as run_ensemble_main
from src.benchmark.run_time_baselines import main as run_time_baselines_main
from src.benchmark.run_time_ensemble import main as run_time_ensemble_main
from src.benchmark.run_time_agentic import main as run_time_agentic_main
from src.benchmark.run_time_agentic_no_validator import main as run_time_agentic_no_validator_main

def main():
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline for TSAD Orchestra")
    parser.add_argument("--baselines", action="store_true", help="Run the baseline algorithms (without agent)")
    parser.add_argument("--ensemble", action="store_true", help="Run the ensemble baseline")
    parser.add_argument("--agentic", action="store_true", help="Run the agentic solution")
    parser.add_argument("--agentic-no-validator", action="store_true", help="Run the agentic solution without the validator (ablation study)")
    parser.add_argument("--time-baselines", action="store_true", help="Measure baseline execution times")
    parser.add_argument("--time-ensemble", action="store_true", help="Measure ensemble execution times")
    parser.add_argument("--time-agentic", action="store_true", help="Measure agentic execution times")
    parser.add_argument("--time-agentic-no-validator", action="store_true", help="Measure agentic without validator execution times")
    parser.add_argument("--all", action="store_true", help="Run baselines, ensemble, agentic, agentic-no-validator, and measure all execution times")

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
        args.all
    ]

    if not any(flags):
        print("Please specify at least one flag: --baselines, --ensemble, --agentic, --agentic-no-validator, --time-baselines, --time-ensemble, --time-agentic, --time-agentic-no-validator, or --all")
        parser.print_help()
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
