# BookApp Semantic API — Mimari

Bu servis **bağımsız bir deploy birimidir**. Flutter (BookApp) ile yalnızca HTTP sözleşmesi üzerinden konuşur; kod paylaşımı yoktur.

## Bağımsızlık durumu

| Bağlantı | Durum |
|----------|--------|
| Flutter kod import | Yok |
| Ortak Python paketi | Yok |
| Paylaşılan `.env` | Yok (kendi `.env`) |
| Supabase şeması | Paylaşılır — `supabase/migrations/` bu repoda |
| HTTP API sözleşmesi | `POST /api/v1/semantic/search`, `POST /api/v1/trbooks/generate-description` |

`bookapp-api/` klasörünü ayrı bir git reposuna kopyalayıp deploy edebilirsiniz.

## Katmanlar (bu serviste)

```
┌─────────────────────────────────────────┐
│  api/          HTTP (FastAPI routers)   │  ← İnce: doğrulama, DTO, status kodları
├─────────────────────────────────────────┤
│  domain/       İş mantığı               │  ← search_service, query_rewriter,
│                                          │    description_generator, normalizer
├─────────────────────────────────────────┤
│  data/         Veri erişimi             │  ← Supabase, Gemini, Google Books
│    datasources/   ham I/O               │
│    repositories/  tablo/RPC soyutlama   │
├─────────────────────────────────────────┤
│  models/       DTO + domain modelleri   │
│  core/         config                   │
└─────────────────────────────────────────┘
```

### Veri katmanı nerede?

**FastAPI reposunda (`app/data/`):**

| Bileşen | Konum | Ne yapar |
|---------|-------|----------|
| Supabase client | `data/datasources/supabase.py` | Bağlantı fabrikası |
| Gemini embeddings | `data/datasources/gemini.py` | Vektör üretimi |
| Google Books | `data/datasources/google_books.py` | Advanced mod ingest |
| Katalog | `data/repositories/book_catalog_repository.py` | `book_catalog` + RPC |
| Sorgu cache | `data/repositories/query_cache_repository.py` | `semantic_query_cache` |
| Arama logu | `data/repositories/search_log_repository.py` | Anonim sunucu logları |

**Flutter reposunda (`lib/features/semantic_discovery/data/`):**

| Bileşen | Ne yapar |
|---------|----------|
| `semantic_api_datasource.dart` | FastAPI HTTP çağrısı (Dio) |
| `semantic_search_log_remote_datasource.dart` | Kullanıcı logları → Supabase |

**Flutter'da olmaması gerekenler:** embedding, pgvector, LLM rewrite, katalog yazma.

**FastAPI'de olmaması gerekenler:** kullanıcı oturumu, `user_books`, favori, UI state.

## Sistem genelinde veri sahipliği

```
Flutter                    FastAPI                    Supabase
───────                    ───────                    ────────
semantic_api_datasource ──HTTP──► search_service ──RPC──► book_catalog
book_identity_cache  ──────────────direct──────────────► book_identity_cache
semantic_search_logs ────────────direct──────────────► semantic_search_logs
google-books edge fn ────────────direct──────────────► (volume detay)
```

- **Semantik arama:** Flutter → FastAPI → Supabase (service_role)
- **Kitap detay / favori:** Flutter → Google Books proxy + Supabase (anon/authenticated)
- **ISBN resolve cache:** Flutter → Supabase (authenticated okuma, RPC ile yazma)

## Taşıma checklist

1. `bookapp-api/` klasörünü yeni repoya kopyala
2. `supabase db push` veya migration'ları hedef projeye uygula
3. `.env` doldur (`GOOGLE_API_KEY`, `SUPABASE_*`, `API_KEY`)
4. `pip install -e .` veya `docker build`
5. Flutter `SEMANTIC_API_BASE_URL` yeni deploy URL'ine güncelle

BookApp monorepo içinde kalsa bile bu katman ayrımı korunmalıdır.
