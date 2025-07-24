import pandas as pd
import psycopg2
import json
from psycopg2.extras import Json
from psycopg2.extras import execute_values

# Database connection settings
db_config = {
    "host": "localhost",
    "dbname": "mind_db",
    "user": "joseandres",
    "password": "",
    "port": 5432
}

# Load datasets

behaviors_path = "/Users/joseandres/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/Thesis/Datasets/MIND/SmallMind/MINDSmall_train/behaviors.tsv"
news_path = "/Users/joseandres/Library/CloudStorage/OneDrive-ScientificNetworkSouthTyrol/Thesis/Datasets/MIND/SmallMind/MINDSmall_train/news.tsv"
news_df = pd.read_csv(news_path, sep="\t", header=None, names=[
    "news_id", "category", "subcategory", "title", "abstract", 
    "url", "title_entities", "abstract_entities"
])

behaviors_df = pd.read_csv(behaviors_path, sep="\t", header=None, names=[
    "impression_id", "user_id", "timestamp", "history", "impressions"
])

def safe_parse_json(x):
    try:
        if pd.isna(x) or x.strip() == "[]":
            return []
        return json.loads(x.replace("'", '"'))
    except Exception:
        return []

news_df['title_entities'] = news_df['title_entities'].apply(safe_parse_json)
news_df['abstract_entities'] = news_df['abstract_entities'].apply(safe_parse_json)

# === Format timestamps ===
behaviors_df['timestamp'] = pd.to_datetime(behaviors_df['timestamp'], errors='coerce')

# === Connect to PostgreSQL ===
conn = psycopg2.connect(**db_config)
cur = conn.cursor()

# === Create tables ===
cur.execute("""
CREATE TABLE IF NOT EXISTS news (
    news_id TEXT PRIMARY KEY,
    category TEXT,
    subcategory TEXT,
    title TEXT,
    abstract TEXT,
    url TEXT,
    title_entities JSONB,
    abstract_entities JSONB
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS behaviors (
    impression_id INTEGER PRIMARY KEY,
    user_id TEXT,
    timestamp TIMESTAMP,
    history TEXT,
    impressions TEXT
);
""")
conn.commit()

# === Insert news data ===
news_values = [
    (
        row.news_id,
        row.category,
        row.subcategory,
        row.title,
        row.abstract,
        row.url,
        Json(row.title_entities),
        Json(row.abstract_entities)
    )
    for row in news_df.itertuples(index=False)
]
execute_values(cur,
    """
    INSERT INTO news (
        news_id, category, subcategory, title, abstract, url, 
        title_entities, abstract_entities
    ) VALUES %s 
    ON CONFLICT (news_id) DO NOTHING
    """,
    news_values
)

# === Insert behaviors data ===
behaviors_values = behaviors_df.values.tolist()
execute_values(cur,
    """
    INSERT INTO behaviors (
        impression_id, user_id, timestamp, history, impressions
    ) VALUES %s 
    ON CONFLICT (impression_id) DO NOTHING
    """,
    behaviors_values
)

# === Finalize ===
conn.commit()
cur.close()
conn.close()
print("Data loaded successfully into PostgreSQL...")