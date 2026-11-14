from typing import List, Optional, Tuple

import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import settings


def get_connection():
    conn = psycopg2.connect(settings.DATABASE_URL)
    register_vector(conn)
    return conn


def store_embedding(gloss_id: int, embedding) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {settings.GLOSSES_TABLE}
                SET embedding = %s, embedding_model = %s
                WHERE id = %s
                """,
                (embedding, settings.MODEL_NAME, gloss_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def find_nearest(
    embedding, top_k: int, exclude_id: Optional[int] = None
) -> List[Tuple[int, str, str, float]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if exclude_id is not None:
                cur.execute(
                    f"""
                    SELECT id, text, language, 1 - (embedding <=> %s) AS similarity
                    FROM {settings.GLOSSES_TABLE}
                    WHERE embedding IS NOT NULL AND id != %s
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, exclude_id, embedding, top_k),
                )
            else:
                cur.execute(
                    f"""
                    SELECT id, text, language, 1 - (embedding <=> %s) AS similarity
                    FROM {settings.GLOSSES_TABLE}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
            return cur.fetchall()
    finally:
        conn.close()
