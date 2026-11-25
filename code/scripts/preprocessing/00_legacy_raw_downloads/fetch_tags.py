"""Fetch the Gamma tags of every market in closed_markets.

Copied from legacy_code/old_thesis_code/python/fetch_tags.py (May 2026) with two changes:
the output table is called closed_market_tags, the name it has in data/preprocessing.duckdb
(the legacy script wrote market_tags), and the database path is an option. Needs aiohttp,
which is not in pyproject.toml. Run after sql/markets/01_load_closed_markets.sql.
"""

import argparse
import asyncio
import aiohttp
import duckdb
from datetime import datetime, timezone

from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parents[3] / "data" / "preprocessing.duckdb")
GAMMA = "https://gamma-api.polymarket.com"

REQUESTS_PER_SECOND = 10
CONCURRENCY = 20
BATCH_SIZE = 1000
MAX_RETRIES = 5


def setup_db(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS closed_market_tags (
            market_id VARCHAR,
            tag_id VARCHAR,
            label VARCHAR,
            slug VARCHAR,
            fetched_at TIMESTAMP
        )
    """)

    con.execute("""
        CREATE TABLE IF NOT EXISTS closed_market_tag_fetch_status (
            market_id VARCHAR PRIMARY KEY,
            status VARCHAR,
            error VARCHAR,
            fetched_at TIMESTAMP
        )
    """)


def get_market_ids_to_fetch(con):
    return [
        row[0]
        for row in con.execute("""
            SELECT DISTINCT CAST(id AS VARCHAR) AS market_id
            FROM closed_markets
            WHERE id IS NOT NULL
              AND CAST(id AS VARCHAR) NOT IN (
                  SELECT market_id
                  FROM closed_market_tag_fetch_status
                  WHERE status = 'OK'
              )
            ORDER BY market_id
        """).fetchall()
    ]


async def fetch_one(session, market_id, limiter):
    url = f"{GAMMA}/markets/{market_id}/tags"

    for attempt in range(1, MAX_RETRIES + 1):
        async with limiter:
            try:
                async with session.get(url, timeout=30) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(2 * attempt)
                        continue

                    if resp.status >= 500:
                        await asyncio.sleep(2 * attempt)
                        continue

                    if resp.status == 404:
                        return market_id, [], "NOT_FOUND", None

                    resp.raise_for_status()
                    data = await resp.json()

                    rows = []
                    now = datetime.now(timezone.utc).isoformat()

                    for tag in data:
                        rows.append((
                            str(market_id),
                            str(tag.get("id")) if tag.get("id") is not None else None,
                            tag.get("label"),
                            tag.get("slug"),
                            now,
                        ))

                    return market_id, rows, "OK", None

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    return market_id, [], "ERROR", str(exc)

                await asyncio.sleep(2 * attempt)


class RateLimiter:
    def __init__(self, rate_per_second):
        self.delay = 1.0 / rate_per_second
        self.lock = asyncio.Lock()
        self.last = 0.0

    async def __aenter__(self):
        async with self.lock:
            now = asyncio.get_running_loop().time()
            wait = self.delay - (now - self.last)
            if wait > 0:
                await asyncio.sleep(wait)
            self.last = asyncio.get_running_loop().time()

    async def __aexit__(self, exc_type, exc, tb):
        pass


async def fetch_batch(market_ids):
    limiter = RateLimiter(REQUESTS_PER_SECOND)

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            fetch_one(session, market_id, limiter)
            for market_id in market_ids
        ]
        return await asyncio.gather(*tasks)


def save_results(con, results):
    tag_rows = []
    status_rows = []

    now = datetime.now(timezone.utc).isoformat()

    for market_id, rows, status, error in results:
        tag_rows.extend(rows)
        status_rows.append((str(market_id), status, error, now))

    if tag_rows:
        con.executemany("""
            INSERT INTO closed_market_tags
            VALUES (?, ?, ?, ?, ?)
        """, tag_rows)

    con.executemany("""
        INSERT OR REPLACE INTO closed_market_tag_fetch_status
        VALUES (?, ?, ?, ?)
    """, status_rows)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DB_PATH)
    args = parser.parse_args()

    con = duckdb.connect(args.db_path)
    setup_db(con)

    market_ids = get_market_ids_to_fetch(con)
    total = len(market_ids)

    print(f"Markets left to fetch: {total:,}")

    for i in range(0, total, BATCH_SIZE):
        batch = market_ids[i:i + BATCH_SIZE]

        print(f"Fetching {i + 1:,} - {i + len(batch):,} / {total:,}")

        results = await fetch_batch(batch)
        save_results(con, results)

        ok = sum(1 for _, _, status, _ in results if status == "OK")
        errors = sum(1 for _, _, status, _ in results if status == "ERROR")

        print(f"Saved batch: OK={ok}, ERROR={errors}")

    con.close()


if __name__ == "__main__":
    asyncio.run(main())