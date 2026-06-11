import argparse
import asyncio

from src.benchmark.run_baselines import main as run_baselines_main
from src.benchmark.agentic import main as run_agentic_main
from src.benchmark.agentic_no_validator import main as run_agentic_no_validator_main
from src.benchmark.ensemble import main as run_ensemble_main

def main():
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline for TSAD Orchestra")
    parser.add_argument("--baselines", action="store_true", help="Run the baseline algorithms (without agent)")
    parser.add_argument("--ensemble", action="store_true", help="Run the ensemble baseline")
    parser.add_argument("--agentic", action="store_true", help="Run the agentic solution")
    parser.add_argument("--agentic-no-validator", action="store_true", help="Run the agentic solution without the validator (ablation study)")
    parser.add_argument("--all", action="store_true", help="Run baselines, agentic, and agentic-no-validator")

    args = parser.parse_args()

    if not any([args.baselines, args.ensemble, args.agentic, args.agentic_no_validator, args.all]):
        print("Please specify at least one flag: --baselines, --ensemble, --agentic, --agentic-no-validator, or --all")
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

if __name__ == "__main__":
    main()
