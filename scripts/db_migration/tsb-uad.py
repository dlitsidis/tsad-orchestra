import os
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


URL = "https://www.thedatum.org/datasets/TSB-AD-U.zip"
# Use Path for robust directory handling across different OS environments
TARGET_DIR = Path("./scripts/db_migration/univariate/TSB-AD-U")
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"  # noqa: E501
SEED = int(os.getenv("RANDOM_SEED", 42))

engine = create_engine(DB_URL)


def download_and_extract():
    print(f"Downloading {URL}...")
    resp = requests.get(URL)
    if resp.status_code == 200:
        with zipfile.ZipFile(BytesIO(resp.content)) as zip_ref:
            zip_ref.extractall(TARGET_DIR.parent)  # Extracts into .../univariate/
        print(f"Extracted to {TARGET_DIR.absolute()}")
        return True
    return False


def migrate_to_db():
    all_files = list(TARGET_DIR.rglob("*.csv"))
    if not all_files:
        print(f"No CSVs found in {TARGET_DIR}")
        return

    # Filter for timeseries with less than 6000 points
    # Assumes 1 header line, so max 6000 lines total (5999 points)
    filtered_files = []
    for f in all_files:
        try:
            with open(f, "r") as file:
                has_less_than_6000 = True
                for i, _ in enumerate(file):
                    if i >= 6000:
                        has_less_than_6000 = False
                        break
                if has_less_than_6000:
                    filtered_files.append(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    datasets = defaultdict(list)
    for f in filtered_files:
        parts = f.name.split("_")
        if len(parts) >= 2:
            datasets[parts[1]].append(f)

    random.seed(SEED)
    selected_files = []
    for files in datasets.values():
        selected_files.extend(random.sample(files, min(len(files), 10)))

    print(f"Migrating {len(selected_files)} files from {len(datasets)} datasets...")

    for i, file_path in enumerate(selected_files, 1):
        # Table names must be lowercase and sanitized
        tbl = file_path.stem.replace("-", "_").lower()
        print(f"[{i}/{len(selected_files)}] Processing {tbl}")

        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        f'CREATE TABLE IF NOT EXISTS "{tbl}" (time TIMESTAMPTZ NOT NULL, data DOUBLE PRECISION, label INTEGER)'  # noqa: E501
                    )
                )  # noqa: E501

                is_hyper = conn.execute(
                    text(
                        "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = :t"  # noqa: E501
                    ),
                    {"t": tbl},
                ).fetchone()  # noqa: E501
                if not is_hyper:
                    conn.execute(text(f"SELECT create_hypertable('\"{tbl}\"', 'time')"))

                df = pd.read_csv(file_path)
                df.columns = df.columns.str.lower()

                if "time" not in df.columns:
                    df["time"] = pd.date_range(start="2026-01-01", periods=len(df), freq="s")

                df.to_sql(
                    tbl, conn, if_exists="append", index=False, method="multi", chunksize=5000
                )  # noqa: E501
            print(f"  Inserted {len(df)} rows")
        except Exception as e:
            print(f"  Error migrating {tbl}: {e}")


if __name__ == "__main__":
    if download_and_extract():
        migrate_to_db()
