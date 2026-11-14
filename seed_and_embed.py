import argparse
import sys

import psycopg2

from app.config import settings
from app.db import get_connection
from app.embedding import embed_texts

SAMPLES = [
    ("thank you", "en"), ("thanks", "en"), ("asante", "sw"),
    ("mother", "en"), ("mama", "sw"), ("father", "en"), ("baba", "sw"),
    ("good morning", "en"), ("habari za asubuhi", "sw"),
    ("hello", "en"), ("hujambo", "sw"),
    ("water", "en"), ("maji", "sw"), ("food", "en"), ("chakula", "sw"),
    ("school", "en"), ("shule", "sw"), ("teacher", "en"), ("mwalimu", "sw"),
    ("friend", "en"), ("rafiki", "sw"), ("please", "en"), ("tafadhali", "sw"),
    ("sorry", "en"), ("samahani", "sw"), ("yes", "en"), ("ndiyo", "sw"),
    ("no", "en"), ("hapana", "sw"), ("help", "en"), ("msaada", "sw"),
]

BATCH = 32


def seed(cur) -> int:
    inserted = 0
    for text, lang in SAMPLES:
        cur.execute(
            f"""
            INSERT INTO {settings.GLOSSES_TABLE} (text, language, status)
            SELECT %s, %s, 'approved'
            WHERE NOT EXISTS (
                SELECT 1 FROM {settings.GLOSSES_TABLE} WHERE text = %s AND language = %s
            )
            """,
            (text, lang, text, lang),
        )
        inserted += cur.rowcount
    return inserted


def embed_missing(cur) -> int:
    cur.execute(
        f"SELECT id, text FROM {settings.GLOSSES_TABLE} WHERE embedding IS NULL ORDER BY id"
    )
    rows = cur.fetchall()
    for start in range(0, len(rows), BATCH):
        chunk = rows[start : start + BATCH]
        vectors = embed_texts([text for _, text in chunk])
        for (row_id, _), vec in zip(chunk, vectors):
            cur.execute(
                f"UPDATE {settings.GLOSSES_TABLE} SET embedding = %s, embedding_model = %s WHERE id = %s",
                (vec, settings.MODEL_NAME, row_id),
            )
        print(f"  embedded {min(start + BATCH, len(rows))}/{len(rows)}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-seed", action="store_true", help="skip inserting sample glosses")
    args = parser.parse_args()

    try:
        conn = get_connection()
    except psycopg2.OperationalError as exc:
        sys.exit(f"Could not connect to the database. Check DATABASE_URL in .env.\n{exc}")

    try:
        with conn.cursor() as cur:
            if not args.no_seed:
                print(f"Seeded {seed(cur)} new sample glosses")
            print("Embedding rows that have no embedding yet...")
            print(f"Done: {embed_missing(cur)} rows embedded")
        conn.commit()
    except psycopg2.errors.UndefinedTable:
        conn.rollback()
        sys.exit(
            f"Table '{settings.GLOSSES_TABLE}' doesn't exist. Run the CREATE TABLE SQL "
            "in the Supabase SQL Editor first."
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
