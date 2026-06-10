import argparse
import asyncio
import sys

from loguru import logger

from src.agent.client import run


async def main():
    parser = argparse.ArgumentParser(description="Run TSAD Orchestra anomaly detection from CLI.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The dataset/series ID or name to analyze (e.g. '1' or 'my_table')",
    )
    args = parser.parse_args()

    print(f"Running TSAD Orchestra anomaly detection on dataset: {args.dataset}")
    
    try:
        report = await run(args.dataset)
        
        print("\n" + "="*50)
        print("DETECTION REPORT")
        print("="*50)
        print(f"Detectors Used: {', '.join(report.detectors_used)}")
        print(f"Total Anomalies: {report.anomaly_count}")
        print("\nSummary:")
        print(report.summary)
        
        if report.anomalies:
            print("\nAnomalies Details:")
            for a in report.anomalies:
                print(f" - Index {a.index}: Value {a.value:.4f} (Severity Score: {a.score:.2f})")
                
    except Exception as e:
        logger.exception("Detection failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
