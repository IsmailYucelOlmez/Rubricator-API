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

### Covers for the Turkish catalog

The Turkish books were imported without covers. Their covers live in bookapp's
`trbooks.image_url`; copy them into the Qdrant payload by ISBN after every
catalog import (only empty `thumbnail_url` fields are filled, vectors are not
touched, so re-running is safe):

```bash
python scripts/backfill_catalog_covers.py --dry-run
python scripts/backfill_catalog_covers.py
python scripts/backfill_catalog_covers.py --undo logs/cover_backfill_<stamp>.jsonl
```

Notes:
- Kitapyurdu cover URLs are rewritten from `wi:100` to `wi:400` (100 px is
  blurry in the app's grid); `http:` becomes `https:`.
- When an ISBN has several `trbooks` rows with different covers, the most
  recently scraped one wins.
- Writes go one `SetPayload` per point, about 1,600 points a minute; a full
  run takes roughly an hour. An interrupted run is resumed by running it again.
- `logs/` is git-ignored: the undo logs stay on the machine that ran the script.
- Search results are not cached with their covers (`semantic_query_cache`
  stores only ISBNs and the rewrite), so new covers show up immediately; no
  API deploy is needed for the data change.

Backfill history:

| Date | Points written | Result | Undo logs |
|---|---|---|---|
| 2026-10-02 | 92,090 (tr 88,606, other 3,484) | Covered points 5,122 → 97,212; 49,746 still have no cover anywhere; a follow-up dry run matched 0 | `cover_backfill_20261002_231356.jsonl` (200, pilot), `..._231833.jsonl` (16,000, interrupted and resumed), `..._233345.jsonl` (75,890) |

## Advanced mode (Faz 2)

Set `GOOGLE_BOOKS_API_KEY` in `.env` for reliable Google Books ingest during advanced searches.

Flutter UI: **Hızlı / Derin** (Quick / Deep) toggle on the semantic tab sends `mode: "advanced"` to the API.

Advanced flow:
1. Gemini rewrites query → Google Books search strings
2. New books normalized → `book_catalog` upsert with embeddings
3. pgvector search (tone is ignored in advanced mode)
4. Results cached in `semantic_query_cache` (Supabase)

## Turkish book descriptions (trbooks)

`POST /api/v1/trbooks/generate-description` — given `{title, author, isbn, language}`,
returns a short, spoiler-free, non-copied Turkish description via Gemini
(`domain/description_generator.py`). Used by the Flutter app's "add a Turkish
book" form (bookapp's trbooks feature) when a user-submitted book has no
description of its own; the result is never persisted server-side, the
Flutter client stores it via the `submit_user_trbook` Supabase RPC.

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
