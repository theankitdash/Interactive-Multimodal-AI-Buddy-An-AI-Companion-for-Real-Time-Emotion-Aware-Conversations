"""Export feedback logs from PostgreSQL as DPO preference pairs.

Usage:
    python -m training.export_feedback --output training/data/dpo_pairs.jsonl
"""
import asyncio
import argparse
import sys
import os

# Add backend to path so we can import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def main(output_path: str, min_interactions: int):
    from utils.db_connect import init_pool, close_pool
    from utils.feedback_collector import feedback_collector

    await init_pool()

    pair_count = await feedback_collector.export_dpo_pairs(
        output_path=output_path,
        min_interactions=min_interactions,
    )

    if pair_count > 0:
        print(f"✓ Exported {pair_count} DPO pairs to {output_path}")
    else:
        print(f"✗ Not enough data yet. Keep collecting interactions.")

    await close_pool()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export feedback logs as DPO pairs")
    parser.add_argument("--output", default="training/data/dpo_pairs.jsonl", help="Output JSONL path")
    parser.add_argument("--min-interactions", type=int, default=500, help="Minimum interactions before export")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    asyncio.run(main(args.output, args.min_interactions))
