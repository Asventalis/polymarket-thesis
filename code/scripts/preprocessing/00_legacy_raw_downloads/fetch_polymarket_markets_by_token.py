#!/usr/bin/env python3

import argparse
import time
from urllib.parse import quote
from typing import Optional, Dict, Tuple

import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


from pathlib import Path

DB_PATH = str(Path(__file__).resolve().parents[3] / "data" / "preprocessing.duckdb")

UNMATCHED_TABLE = "unmatched_token_ids"
MARKETS_TABLE = "markets_fetched_from_token_transactions"
FAILURES_TABLE = "markets_by_token_fetch_failures"

BASE_URL = "https://clob.polymarket.com/markets-by-token"


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        backoff_factor=1.0,
        respect_retry_after_header=True,
    )

    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "polymarket-token-market-fetcher/1.0",
    })

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {qident(MARKETS_TABLE)} (
            condition_id VARCHAR PRIMARY KEY,
            primary_token_id VARCHAR NOT NULL,
            secondary_token_id VARCHAR NOT NULL,
            fetched_from_token_id VARCHAR NOT NULL,
            fetched_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {qident(FAILURES_TABLE)} (
            token_id VARCHAR PRIMARY KEY,
            status_code INTEGER,
            error_text VARCHAR,
            attempted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Helpful but not strictly required.
    try:
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_unmatched_token_id "
            f"ON {qident(UNMATCHED_TABLE)}(token_id)"
        )
    except Exception:
        pass


def load_pending_tokens(
    con: duckdb.DuckDBPyConnection,
    limit: Optional[int],
) -> list[str]:
    sql = f"""
        SELECT DISTINCT CAST(token_id AS VARCHAR) AS token_id
        FROM {qident(UNMATCHED_TABLE)}
        WHERE token_id IS NOT NULL
    """

    if limit is not None:
        sql += f"\nLIMIT {int(limit)}"

    rows = con.execute(sql).fetchall()
    return [row[0] for row in rows]


def fetch_market(
    session: requests.Session,
    token_id: str,
    timeout: float,
) -> Tuple[Optional[Dict[str, str]], Optional[int], Optional[str]]:
    url = f"{BASE_URL}/{quote(token_id, safe='')}"

    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return None, None, str(exc)

    if response.status_code != 200:
        return None, response.status_code, response.text[:1000]

    try:
        data = response.json()
    except ValueError as exc:
        return None, response.status_code, f"Invalid JSON: {exc}; body={response.text[:1000]}"

    condition_id = str(data.get("condition_id") or "").strip()
    primary_token_id = str(data.get("primary_token_id") or "").strip()
    secondary_token_id = str(data.get("secondary_token_id") or "").strip()

    if not condition_id or not primary_token_id or not secondary_token_id:
        return None, response.status_code, f"Missing expected fields: {data}"

    if token_id not in {primary_token_id, secondary_token_id}:
        return None, response.status_code, (
            f"API response did not contain requested token_id={token_id}; response={data}"
        )

    return {
        "condition_id": condition_id,
        "primary_token_id": primary_token_id,
        "secondary_token_id": secondary_token_id,
    }, response.status_code, None


def save_failure(
    con: duckdb.DuckDBPyConnection,
    token_id: str,
    status_code: Optional[int],
    error_text: str,
) -> None:
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            f"DELETE FROM {qident(FAILURES_TABLE)} WHERE token_id = ?",
            [token_id],
        )
        con.execute(
            f"""
            INSERT INTO {qident(FAILURES_TABLE)}
                (token_id, status_code, error_text, attempted_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [token_id, status_code, error_text],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def save_market_and_remove_tokens(
    con: duckdb.DuckDBPyConnection,
    fetched_from_token_id: str,
    market: Dict[str, str],
) -> None:
    condition_id = market["condition_id"]
    primary_token_id = market["primary_token_id"]
    secondary_token_id = market["secondary_token_id"]

    con.execute("BEGIN TRANSACTION")
    try:
        # One row per market / condition_id.
        con.execute(
            f"""
            INSERT INTO {qident(MARKETS_TABLE)}
                (
                    condition_id,
                    primary_token_id,
                    secondary_token_id,
                    fetched_from_token_id,
                    fetched_at
                )
            SELECT ?, ?, ?, ?, CURRENT_TIMESTAMP
            WHERE NOT EXISTS (
                SELECT 1
                FROM {qident(MARKETS_TABLE)}
                WHERE condition_id = ?
            )
            """,
            [
                condition_id,
                primary_token_id,
                secondary_token_id,
                fetched_from_token_id,
                condition_id,
            ],
        )

        # Remove both tokens belonging to the market from unmatched_token_ids.
        con.execute(
            f"""
            DELETE FROM {qident(UNMATCHED_TABLE)}
            WHERE CAST(token_id AS VARCHAR) IN (?, ?)
            """,
            [primary_token_id, secondary_token_id],
        )

        # Clear old failure records for both tokens, if any.
        con.execute(
            f"""
            DELETE FROM {qident(FAILURES_TABLE)}
            WHERE token_id IN (?, ?)
            """,
            [primary_token_id, secondary_token_id],
        )

        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    con = duckdb.connect(args.db_path)
    session = make_session()

    init_db(con)

    tokens = load_pending_tokens(con, args.limit)
    pending = set(tokens)

    print(f"Loaded {len(tokens):,} distinct unmatched token_ids")

    fetched_markets = 0
    failed_tokens = 0
    skipped_counterpart_tokens = 0

    for i, token_id in enumerate(tokens, start=1):
        # This token may already have been removed because its counterpart
        # was fetched earlier in the loop.
        if token_id not in pending:
            skipped_counterpart_tokens += 1
            continue

        market, status_code, error_text = fetch_market(
            session=session,
            token_id=token_id,
            timeout=args.timeout,
        )

        if market is None:
            failed_tokens += 1
            save_failure(
                con=con,
                token_id=token_id,
                status_code=status_code,
                error_text=error_text or "Unknown error",
            )
        else:
            save_market_and_remove_tokens(
                con=con,
                fetched_from_token_id=token_id,
                market=market,
            )

            fetched_markets += 1

            pending.discard(market["primary_token_id"])
            pending.discard(market["secondary_token_id"])

        if args.delay > 0:
            time.sleep(args.delay)

        if i % args.progress_every == 0:
            remaining_in_db = con.execute(
                f"SELECT COUNT(*) FROM {qident(UNMATCHED_TABLE)}"
            ).fetchone()[0]

            print(
                f"Processed={i:,} | "
                f"Markets fetched={fetched_markets:,} | "
                f"Failures={failed_tokens:,} | "
                f"Skipped counterparts={skipped_counterpart_tokens:,} | "
                f"Remaining unmatched in DB={remaining_in_db:,}"
            )

    remaining_in_db = con.execute(
        f"SELECT COUNT(*) FROM {qident(UNMATCHED_TABLE)}"
    ).fetchone()[0]

    total_markets = con.execute(
        f"SELECT COUNT(*) FROM {qident(MARKETS_TABLE)}"
    ).fetchone()[0]

    total_failures = con.execute(
        f"SELECT COUNT(*) FROM {qident(FAILURES_TABLE)}"
    ).fetchone()[0]

    con.execute("CHECKPOINT")
    con.close()

    print("Done.")
    print(f"Fetched markets this run: {fetched_markets:,}")
    print(f"Skipped counterpart tokens this run: {skipped_counterpart_tokens:,}")
    print(f"Failed tokens this run: {failed_tokens:,}")
    print(f"Total rows in {MARKETS_TABLE}: {total_markets:,}")
    print(f"Total rows in {FAILURES_TABLE}: {total_failures:,}")
    print(f"Remaining rows in {UNMATCHED_TABLE}: {remaining_in_db:,}")


if __name__ == "__main__":
    main()