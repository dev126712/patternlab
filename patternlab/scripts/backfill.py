"""Backfill daily session fingerprints from OANDA history into Postgres.

Fetches N days of M1 candles via the official OANDA v20 API (paginated), slices them into
UTC calendar-day sessions, and upserts each day's fingerprint + analysis through
repository.save_session. Re-running is safe: existing (instrument, date) rows are updated,
not duplicated — which is also what makes this the nightly refresh job.

The in-progress UTC day is skipped: a partial session in the match library would be noise
(and it's what live queries compare *against*).

Usage (Postgres is only reachable on the compose network):
    docker compose exec api uv run python scripts/backfill.py --days 90
Nightly refresh (cron):
    docker compose exec -T api uv run python scripts/backfill.py --days 3
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt

from patternlab.config import params, settings
from patternlab.data.loader import fetch_oanda
from patternlab.storage import repository
from patternlab.storage.db import get_database

# Below this, a calendar day is a weekend stub / holiday, not a tradeable session.
MIN_BARS_PER_SESSION = 300


async def backfill(days: int, instrument: str, granularity: str, label: str) -> None:
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not configured")
    if not settings.oanda_api_token:
        raise SystemExit("OANDA_API_TOKEN is not configured")

    count = days * 1440
    print(f"Fetching ~{count} {granularity} bars of {instrument} from OANDA…")
    df = fetch_oanda(instrument, granularity, count)
    print(f"Got {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    database = get_database()
    await database.init_models()

    # Both boundary days of the fetch window are unusable: the last is still in progress,
    # and the first starts mid-day (count-based fetch), so its stub would clobber a full
    # session saved by an earlier, deeper run.
    in_progress = df.index[-1].date()
    truncated = df.index[0].date()
    saved = skipped = 0
    async with database.session() as session:
        for day, day_df in df.groupby(df.index.date):
            if day in (in_progress, truncated) or len(day_df) < MIN_BARS_PER_SESSION:
                skipped += 1
                continue
            record = await repository.save_session(session, day_df, label, day, params)
            saved += 1
            print(f"  {day}  {len(day_df):>5} bars  -> session #{record.id}")
    print(f"Done: {saved} sessions saved/updated, {skipped} skipped "
          f"(in-progress day + short/weekend days)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--instrument", default="XAU_USD", help="OANDA instrument name")
    ap.add_argument("--granularity", default="M1")
    ap.add_argument("--label", default=None,
                    help="Stored instrument label (default: OANDA:<instrument without underscore>)")
    args = ap.parse_args()
    label = args.label or f"OANDA:{args.instrument.replace('_', '')}"
    asyncio.run(backfill(args.days, args.instrument, args.granularity, label))


if __name__ == "__main__":
    main()
