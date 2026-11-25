#!/usr/bin/env python3

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DB_PATH = str(Path(__file__).resolve().parents[3] / "data" / "preprocessing.duckdb")

SOURCE_TABLE = "markets_missing"
TARGET_TABLE = "missing_fetched_markets"
CLOB_TARGET_TABLE = "missing_fetched_clob_markets"
STATUS_TABLE = "missing_fetched_markets_status"

GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
CLOB_MARKET_URL = "https://clob.polymarket.com/clob-markets"


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        "User-Agent": "polymarket-missing-markets-scraper/1.1",
    })

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def init_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {qident(STATUS_TABLE)} (
            condition_id VARCHAR PRIMARY KEY,
            status VARCHAR NOT NULL, -- fetched | fetched_clob_only | not_found | error
            jsonl_file_path VARCHAR,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_http_status INTEGER,
            last_error VARCHAR,
            last_attempt_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            fetched_at TIMESTAMP
        )
    """)

    try:
        con.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{STATUS_TABLE}_status "
            f"ON {qident(STATUS_TABLE)}(status)"
        )
    except Exception:
        pass


def load_next_batch(
    con: duckdb.DuckDBPyConnection,
    batch_size: int,
    max_attempts: int,
    retry_not_found: bool,
) -> List[str]:
    query = f"""
        WITH missing AS (
            SELECT DISTINCT LOWER(TRIM(CAST(market_id AS VARCHAR))) AS condition_id
            FROM {qident(SOURCE_TABLE)}
            WHERE market_id IS NOT NULL
              AND TRIM(CAST(market_id AS VARCHAR)) <> ''
        )
        SELECT m.condition_id
        FROM missing m
        LEFT JOIN {qident(STATUS_TABLE)} s
            ON s.condition_id = m.condition_id
        WHERE
            s.condition_id IS NULL
            OR (
                s.status = 'error'
                AND s.attempts < ?
            )
            OR (
                ? = TRUE
                AND s.status = 'not_found'
                AND s.attempts < ?
            )
        ORDER BY m.condition_id
        LIMIT ?
    """

    return [
        row[0]
        for row in con.execute(
            query,
            [max_attempts, retry_not_found, max_attempts, batch_size],
        ).fetchall()
    ]


def fetch_gamma_markets_for_conditions(
    session: requests.Session,
    condition_ids: List[str],
    closed: bool,
    timeout: float,
) -> Tuple[List[Dict[str, Any]], Optional[int], Optional[str]]:
    params = [
        ("closed", str(closed).lower()),
        ("include_tag", "true"),
        ("limit", str(max(len(condition_ids), 1))),
    ]

    # Gamma array encoding:
    # condition_ids=0x...&condition_ids=0x...
    for condition_id in condition_ids:
        params.append(("condition_ids", condition_id))

    try:
        response = session.get(
            GAMMA_MARKETS_URL,
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return [], None, str(exc)

    if response.status_code != 200:
        return [], response.status_code, response.text[:2000]

    try:
        payload = response.json()
    except ValueError as exc:
        return [], response.status_code, f"Invalid JSON: {exc}; body={response.text[:2000]}"

    if not isinstance(payload, list):
        return [], response.status_code, f"Unexpected Gamma response shape: {type(payload)}"

    requested = {x.lower() for x in condition_ids}

    markets: List[Dict[str, Any]] = []
    for market in payload:
        if not isinstance(market, dict):
            continue

        condition_id = str(market.get("conditionId", "")).strip().lower()

        # Safety filter: only keep markets from this batch.
        if condition_id not in requested:
            continue

        enriched = dict(market)
        enriched["_scraped_condition_id"] = condition_id
        enriched["_source_api"] = "gamma"
        enriched["_scrape_closed_param"] = closed
        enriched["_scraped_at_utc"] = now_utc_iso()

        markets.append(enriched)

    return markets, response.status_code, None


def fetch_clob_market_for_condition(
    session: requests.Session,
    condition_id: str,
    timeout: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    try:
        response = session.get(
            f"{CLOB_MARKET_URL}/{condition_id}",
            timeout=timeout,
        )
    except requests.RequestException as exc:
        return None, None, str(exc)

    if response.status_code != 200:
        return None, response.status_code, response.text[:2000]

    try:
        payload = response.json()
    except ValueError as exc:
        return None, response.status_code, f"Invalid JSON: {exc}; body={response.text[:2000]}"

    if not isinstance(payload, dict):
        return None, response.status_code, f"Unexpected CLOB response shape: {type(payload)}"

    returned_condition_id = str(payload.get("c", "")).strip().lower()

    if returned_condition_id != condition_id.lower():
        return None, response.status_code, (
            f"CLOB response condition mismatch: requested={condition_id}, returned={returned_condition_id}"
        )

    enriched = dict(payload)
    enriched["_scraped_condition_id"] = condition_id.lower()
    enriched["_source_api"] = "clob"
    enriched["_scraped_at_utc"] = now_utc_iso()

    return enriched, response.status_code, None


def write_jsonl(
    markets: List[Dict[str, Any]],
    raw_dir: Path,
    batch_no: int,
    prefix: str,
) -> Optional[Path]:
    if not markets:
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = raw_dir / f"{prefix}_batch_{batch_no:06d}_{timestamp}_{len(markets)}.jsonl"

    with path.open("w", encoding="utf-8") as f:
        for market in markets:
            f.write(json.dumps(market, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

    return path


def rebuild_gamma_target_table_from_jsonl(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> None:
    files = sorted(str(path) for path in raw_dir.glob("*.jsonl"))

    if not files:
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE {qident(TARGET_TABLE)} AS
        WITH src AS (
            SELECT *
            FROM read_json_auto(
                ?,
                format = 'newline_delimited',
                union_by_name = true,
                sample_size = -1,
                timestampformat = 'iso'
            )
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY LOWER(CAST(conditionId AS VARCHAR))
                    ORDER BY TRY_CAST(_scraped_at_utc AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM src
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
    """, [files])

    try:
        con.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{TARGET_TABLE}_condition_id
            ON {qident(TARGET_TABLE)}(conditionId)
        """)
    except Exception:
        pass


def rebuild_clob_target_table_from_jsonl(
    con: duckdb.DuckDBPyConnection,
    raw_dir: Path,
) -> None:
    files = sorted(str(path) for path in raw_dir.glob("*.jsonl"))

    if not files:
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE {qident(CLOB_TARGET_TABLE)} AS
        WITH src AS (
            SELECT *
            FROM read_json_auto(
                ?,
                format = 'newline_delimited',
                union_by_name = true,
                sample_size = -1,
                timestampformat = 'iso'
            )
        ),
        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY LOWER(CAST(c AS VARCHAR))
                    ORDER BY TRY_CAST(_scraped_at_utc AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM src
        )
        SELECT * EXCLUDE (rn)
        FROM ranked
        WHERE rn = 1
    """, [files])

    try:
        con.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{CLOB_TARGET_TABLE}_condition_id
            ON {qident(CLOB_TARGET_TABLE)}(c)
        """)
    except Exception:
        pass


def get_previous_attempts(con: duckdb.DuckDBPyConnection, condition_id: str) -> int:
    row = con.execute(f"""
        SELECT attempts
        FROM {qident(STATUS_TABLE)}
        WHERE condition_id = ?
    """, [condition_id]).fetchone()

    return int(row[0]) if row else 0


def update_status(
    con: duckdb.DuckDBPyConnection,
    batch_condition_ids: List[str],
    gamma_fetched_condition_ids: Set[str],
    clob_fetched_condition_ids: Set[str],
    gamma_jsonl_path: Optional[Path],
    clob_jsonl_path: Optional[Path],
    gamma_http_status: Optional[int],
    gamma_error: Optional[str],
    clob_errors: Dict[str, str],
) -> None:
    con.execute("BEGIN TRANSACTION")

    try:
        for condition_id in batch_condition_ids:
            condition_id = condition_id.lower()
            attempts = get_previous_attempts(con, condition_id) + 1

            if gamma_error is not None:
                status = "error"
                fetched_at = None
                file_path = None
                last_http_status = gamma_http_status
                last_error = gamma_error

            elif condition_id in gamma_fetched_condition_ids:
                status = "fetched"
                fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file_path = str(gamma_jsonl_path) if gamma_jsonl_path else None
                last_http_status = 200
                last_error = None

            elif condition_id in clob_fetched_condition_ids:
                status = "fetched_clob_only"
                fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file_path = str(clob_jsonl_path) if clob_jsonl_path else None
                last_http_status = 200
                last_error = "Not found in Gamma; fetched from CLOB fallback"

            else:
                status = "not_found"
                fetched_at = None
                file_path = None
                last_http_status = None
                last_error = clob_errors.get(condition_id)

            con.execute(f"""
                DELETE FROM {qident(STATUS_TABLE)}
                WHERE condition_id = ?
            """, [condition_id])

            con.execute(f"""
                INSERT INTO {qident(STATUS_TABLE)} (
                    condition_id,
                    status,
                    jsonl_file_path,
                    attempts,
                    last_http_status,
                    last_error,
                    last_attempt_at,
                    fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, [
                condition_id,
                status,
                file_path,
                attempts,
                last_http_status,
                last_error,
                fetched_at,
            ])

        con.execute("COMMIT")

    except Exception:
        con.execute("ROLLBACK")
        raise


def status_summary(con: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    rows = con.execute(f"""
        SELECT status, COUNT(*) AS n
        FROM {qident(STATUS_TABLE)}
        GROUP BY status
        ORDER BY status
    """).fetchall()

    return {str(status): int(n) for status, n in rows}


def count_table_rows(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    try:
        return int(con.execute(
            f"SELECT COUNT(*) FROM {qident(table_name)}"
        ).fetchone()[0])
    except Exception:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-not-found", action="store_true")
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Defaults to <db_dir>/missing_fetched_markets_jsonl",
    )
    parser.add_argument(
        "--clob-raw-dir",
        default=None,
        help="Defaults to <db_dir>/missing_fetched_clob_markets_jsonl",
    )

    args = parser.parse_args()

    db_path = Path(args.db_path)
    raw_dir = Path(args.raw_dir) if args.raw_dir else db_path.parent / "missing_fetched_markets_jsonl"
    clob_raw_dir = (
        Path(args.clob_raw_dir)
        if args.clob_raw_dir
        else db_path.parent / "missing_fetched_clob_markets_jsonl"
    )

    con = duckdb.connect(str(db_path))
    session = make_session()

    init_db(con)

    batch_no = 0

    while True:
        if args.max_batches is not None and batch_no >= args.max_batches:
            break

        condition_ids = load_next_batch(
            con=con,
            batch_size=args.batch_size,
            max_attempts=args.max_attempts,
            retry_not_found=args.retry_not_found,
        )

        if not condition_ids:
            break

        batch_no += 1
        print(f"\nBatch {batch_no}: {len(condition_ids):,} conditionIds")

        gamma_markets: List[Dict[str, Any]] = []
        gamma_error: Optional[str] = None
        gamma_http_status: Optional[int] = None

        for closed in (False, True):
            markets, http_status, error = fetch_gamma_markets_for_conditions(
                session=session,
                condition_ids=condition_ids,
                closed=closed,
                timeout=args.timeout,
            )

            gamma_http_status = http_status

            if error is not None:
                gamma_error = error
                print(f"  Gamma closed={closed}: ERROR {error[:300]}")
                break

            print(f"  Gamma closed={closed}: fetched {len(markets):,} markets")
            gamma_markets.extend(markets)

            if args.delay > 0:
                time.sleep(args.delay)

        # Deduplicate Gamma rows across closed=false and closed=true.
        gamma_deduped: Dict[str, Dict[str, Any]] = {}
        for market in gamma_markets:
            cid = str(market.get("conditionId", "")).strip().lower()
            if cid:
                gamma_deduped[cid] = market

        gamma_markets = list(gamma_deduped.values())
        gamma_fetched_condition_ids = set(gamma_deduped.keys())

        gamma_jsonl_path = write_jsonl(
            markets=gamma_markets,
            raw_dir=raw_dir,
            batch_no=batch_no,
            prefix="missing_gamma_markets",
        )

        if gamma_jsonl_path is not None:
            rebuild_gamma_target_table_from_jsonl(con, raw_dir=raw_dir)

        clob_markets: List[Dict[str, Any]] = []
        clob_fetched_condition_ids: Set[str] = set()
        clob_errors: Dict[str, str] = {}

        # CLOB fallback only for conditionIds not found in Gamma.
        if gamma_error is None:
            gamma_not_found_ids = sorted(set(condition_ids) - gamma_fetched_condition_ids)

            if gamma_not_found_ids:
                print(f"  CLOB fallback candidates: {len(gamma_not_found_ids):,}")

            for condition_id in gamma_not_found_ids:
                clob_market, clob_http_status, clob_error = fetch_clob_market_for_condition(
                    session=session,
                    condition_id=condition_id,
                    timeout=args.timeout,
                )

                if clob_market is not None:
                    clob_markets.append(clob_market)
                    clob_fetched_condition_ids.add(condition_id.lower())
                    print(f"    CLOB fetched: {condition_id}")
                else:
                    clob_errors[condition_id.lower()] = clob_error or f"HTTP {clob_http_status}"
                    print(
                        f"    CLOB not found/error: {condition_id}: "
                        f"{clob_errors[condition_id.lower()][:200]}"
                    )

                if args.delay > 0:
                    time.sleep(args.delay)

        clob_jsonl_path = write_jsonl(
            markets=clob_markets,
            raw_dir=clob_raw_dir,
            batch_no=batch_no,
            prefix="missing_clob_markets",
        )

        if clob_jsonl_path is not None:
            rebuild_clob_target_table_from_jsonl(con, raw_dir=clob_raw_dir)

        update_status(
            con=con,
            batch_condition_ids=condition_ids,
            gamma_fetched_condition_ids=gamma_fetched_condition_ids,
            clob_fetched_condition_ids=clob_fetched_condition_ids,
            gamma_jsonl_path=gamma_jsonl_path,
            clob_jsonl_path=clob_jsonl_path,
            gamma_http_status=gamma_http_status,
            gamma_error=gamma_error,
            clob_errors=clob_errors,
        )

        print(f"Batch {batch_no} done.")
        print(f"  Gamma JSONL file: {gamma_jsonl_path}")
        print(f"  CLOB JSONL file: {clob_jsonl_path}")
        print(f"  Gamma fetched in batch: {len(gamma_markets):,}")
        print(f"  CLOB-only fetched in batch: {len(clob_markets):,}")
        print(f"  {TARGET_TABLE} rows: {count_table_rows(con, TARGET_TABLE):,}")
        print(f"  {CLOB_TARGET_TABLE} rows: {count_table_rows(con, CLOB_TARGET_TABLE):,}")
        print(f"  status: {status_summary(con)}")

        con.execute("CHECKPOINT")

    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()