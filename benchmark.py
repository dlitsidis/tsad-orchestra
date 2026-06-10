import argparse
import asyncio

from src.benchmark.run_baselines import main as run_baselines_main
from src.benchmark.agentic import main as run_agentic_main

def main():
    parser = argparse.ArgumentParser(description="Run benchmarking pipeline for TSAD Orchestra")
    parser.add_argument("--baselines", action="store_true", help="Run the baseline algorithms (without agent)")
    parser.add_argument("--agentic", action="store_true", help="Run the agentic solution")
    parser.add_argument("--all", action="store_true", help="Run both baselines and agentic solution")

    args = parser.parse_args()

    if not any([args.baselines, args.agentic, args.all]):
        print("Please specify at least one flag: --baselines, --agentic, or --all")
        parser.print_help()
        return

    if args.baselines or args.all:
        print("\n=== Running Baselines ===")
        run_baselines_main()

    if args.agentic or args.all:
        print("\n=== Running Agentic Solution ===")
        asyncio.run(run_agentic_main())

if __name__ == "__main__":
    main()
