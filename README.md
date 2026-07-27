# BookApp Semantic API

FastAPI service for semantic book discovery (Gemini embeddings + Supabase pgvector).

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for layer boundaries and standalone deployment.

## Setup

```bash
cd bookapp-api
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # fill GOOGLE_API_KEY, SUPABASE_*, API_KEY
```

## Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

## Import catalog

Copy `books_with_emotions.csv` from the source recommender project, then:

```bash
python scripts/build_embeddings.py --csv path/to/books_with_emotions.csv --batch-size 50 --resume
```

After import, rebuild the IVFFlat index in Supabase SQL:

```sql
reindex index book_catalog_embedding_idx;
analyze book_catalog;
```

## Advanced mode (Faz 2)

Set `GOOGLE_BOOKS_API_KEY` in `.env` for reliable Google Books ingest during advanced searches.

Flutter UI: **Hızlı / Derin** (Quick / Deep) toggle on the semantic tab sends `mode: "advanced"` to the API.

Advanced flow:
1. Gemini rewrites query → Google Books search strings
2. New books normalized → `book_catalog` upsert with embeddings
3. pgvector search (tone is ignored in advanced mode)
4. Results cached in `semantic_query_cache` (Supabase)

## Analytics

- Flutter logs authenticated searches to `semantic_search_logs`
- FastAPI logs anonymous searches (user_id null) server-side

## Flutter env

Add to `env.development.json`:

```json
"SEMANTIC_API_BASE_URL": "http://localhost:8000",
"SEMANTIC_API_KEY": "your-api-key"
```

Android emulator: use `http://10.0.2.2:8000` instead of `localhost`.
