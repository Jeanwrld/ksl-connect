import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
    )
    MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    DUPLICATE_THRESHOLD: float = float(os.getenv("DUPLICATE_THRESHOLD", "0.88"))
    TOP_K_DEFAULT: int = int(os.getenv("TOP_K_DEFAULT", "5"))
    GLOSSES_TABLE: str = os.getenv("GLOSSES_TABLE", "glosses")
    API_KEY: str = os.getenv("AI_SERVICE_API_KEY", "")


settings = Settings()
