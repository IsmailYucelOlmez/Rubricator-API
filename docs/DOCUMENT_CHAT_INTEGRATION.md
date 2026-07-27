# Belge Sohbeti (PDF / EPUB) — Entegrasyon Spesifikasyonu

Bu doküman, [ChatPDF](https://github.com/ArmaanSeth/ChatPDF) tarzı RAG sohbet özelliğinin **bookapp-api** reposuna nasıl entegre edileceğini tanımlar. Mevcut semantik kitap araması (`POST /api/v1/semantic/search`) ile **paralel**, bağımsız bir modüldür.

## Özet kararlar

| Konu | Karar |
|------|--------|
| Oturum modeli | **B) Ephemeral server session** — sunucu tarafında TTL’li bellek/Redis |
| Orijinal dosya | **Saklanmaz** (işleme sonrası atılır) |
| Chunk / embedding | Yalnızca oturum süresince (kalıcı DB yok) |
| Chat geçmişi | **Kalıcı kayıt yok**; takip soruları için yalnızca oturum belleğinde son N tur |
| Desteklenen formatlar | **PDF** + **EPUB** |
| LLM / embedding | Mevcut **Gemini** altyapısı (`GeminiEmbeddingClient`, `ChatGoogleGenerativeAI`) |
| Vector store | Oturum içi in-memory / Redis (pgvector **kullanılmaz**) |
| Kalıcı Supabase tablosu | **Gerekmez** (bu özellik için yeni migration yok) |

---

## 1. Amaç ve kapsam

Kullanıcı kendi yüklediği kitap dosyası (PDF veya EPUB) ile doğal dilde soru-cevap yapar. Sunucu:

1. Dosyayı doğrular ve metin çıkarır
2. Metni parçalara (chunk) böler
3. Gemini ile embedding üretir
4. Oturum kimliği (`sessionId`) döner
5. Her soruda ilgili chunk’ları getirip (RAG) Gemini ile cevap üretir

**Kapsam dışı:**

- Kalıcı belge kütüphanesi / kullanıcı dosya arşivi
- Chat geçmişinin DB’ye yazılması
- DRM korumalı EPUB (Kindle vb.)
- Mevcut `book_catalog` semantik aramasının değiştirilmesi

---

## 2. Mevcut mimari ile ilişki

```
┌─────────────────────────────────────────────────────────────────┐
│                        bookapp-api (FastAPI)                     │
├──────────────────────────────┬──────────────────────────────────┤
│  Mevcut: semantic modülü      │  Yeni: document-chat modülü       │
│  POST /semantic/search        │  POST /sessions                  │
│  Kalıcı: book_catalog         │  POST /sessions/{id}/chat        │
│  pgvector (Supabase)          │  Ephemeral session store         │
│  Kitap önerisi                │  Kullanıcı belgesi Q&A           │
└──────────────────────────────┴──────────────────────────────────┘
```

İki modül **aynı** bileşenleri paylaşır:

- `app/core/config.py` — limit ve model ayarları
- `app/data/datasources/gemini.py` — `GeminiEmbeddingClient`, `embed_with_retry`
- `app/api/routers/semantic.py` — `_verify_api_key` pattern’i
- `GOOGLE_API_KEY`, `API_KEY` ortam değişkenleri

**Paylaşılmayan / karıştırılmayan:**

- `book_catalog` embedding’leri (768 boyut, kitap metadata) ile oturum chunk’ları aynı tabloda **tutulmaz**
- ChatPDF’den gelen FAISS, HuggingFace, OpenAI, Streamlit **kullanılmaz**

Detaylı katman yapısı için: [ARCHITECTURE.md](../ARCHITECTURE.md).

---

## 3. ChatPDF’den alınacak ve çıkarılacaklar

### Alınacak (uyarlanmış iş mantığı)

| ChatPDF | bookapp-api karşılığı |
|---------|------------------------|
| `get_pdf_text()` | `app/domain/document_extractors/pdf_extractor.py` |
| `get_chunks()` | `app/domain/document_chunker.py` |
| `get_vectorstore()` | `app/data/session_store/` — oturum içi vektör arama |
| `get_conversationchain()` | `app/domain/document_chat_service.py` — RAG + condense prompt |
| Condense question prompt | `CONDENSE_QUESTION_PROMPT` sabiti |

### Çıkarılacak

| Bileşen | Neden |
|---------|--------|
| `streamlit` | FastAPI zaten HTTP API sağlıyor |
| `htmlTemplates.py` | UI client’ta (Flutter) |
| `st.session_state` | `SessionStore` + TTL |
| FAISS (kalıcı) | Oturum bitince silinen ephemeral store |
| `HuggingFaceEmbeddings` | Gemini embedding (768d) |
| `ChatOpenAI` | `ChatGoogleGenerativeAI` |
| `langchain==0.0.339` import’ları | Güncel `langchain-core` / `langchain-text-splitters` |
| ChatPDF `requirements.txt` tamamı | torch, transformers, streamlit vb. gereksiz |

---

## 4. Ephemeral server session (B)

### Akış

```
Client                          FastAPI                         SessionStore
  │                                │                                  │
  │── POST /sessions (PDF/EPUB) ──►│ validate → create pending        │
  │                                │─────────────────────────────────►│ put(status=processing)
  │◄── { sessionId, status:        │                                  │
  │      processing } ─────────────│                                  │
  │                                │ background: extract→chunk→embed  │
  │                                │   (parallel embedding batches)   │
  │                                │─────────────────────────────────►│ put(status=ready)
  │── GET /sessions/{id} ─────────►│                                  │
  │◄── { status, chunksEmbedded } ─│                                  │
  │                                │                                  │
  │── POST /sessions/{id}/chat ───►│ get session → RAG → Gemini       │
  │                                │ append ephemeral chat turn       │
  │                                │─────────────────────────────────►│ touch TTL
  │◄── { answer, sources } ────────│                                  │
  │                                │                                  │
  │   (TTL dolunca veya DELETE)    │                                  │
  │                                │◄─────────────────────────────────│ expire → delete all
```

### Oturum verisi (bellek / Redis)

```python
@dataclass
class DocumentSession:
    session_id: str
    format: Literal["pdf", "epub"]
    filename: str
    created_at: datetime
    expires_at: datetime
    last_accessed_at: datetime
    page_count: int | None          # PDF; EPUB için None
    chapter_count: int | None       # EPUB; PDF için None
    word_count: int
    chunk_count: int
    chunks: list[DocumentChunk]       # content + embedding vector
    chat_turns: list[ChatTurn]        # yalnızca oturum belleğinde
    question_count: int
    status: Literal["processing", "ready", "failed"]
    error_message: str | None
    chunks_total: int
    chunks_embedded: int
```

```python
@dataclass
class DocumentChunk:
    index: int
    content: str
    embedding: list[float]            # 768 boyut (Gemini)
    metadata: dict                    # page, chapter_title, source_range
```

```python
@dataclass
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
```

### TTL politikası

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `session_idle_ttl_minutes` | 45 | Son istekten bu süre sonra oturum silinir (sliding window) |
| `session_max_ttl_minutes` | 120 | Oluşturulmadan itibaren mutlak üst sınır |
| `session_cleanup_interval_seconds` | 60 | Arka plan süpürme (in-memory store için) |

Her başarılı `/chat` veya `GET /sessions/{id}/status` isteği `last_accessed_at` ve `expires_at` günceller.

### Store implementasyonları

| Ortam | Sınıf | Konum |
|-------|-------|-------|
| MVP / tek instance | `InMemorySessionStore` | `app/data/session_store/memory.py` |
| Production / çok instance | `RedisSessionStore` | `app/data/session_store/redis.py` |

**Not:** Çok instance deploy’da in-memory store oturumları instance’lar arasında paylaşmaz; production’da Redis şarttır.

### Chat geçmişi (kalıcı değil)

- Takip soruları (“bir önceki gibi…”) için sunucu oturumda **son 4 tur** (2 user + 2 assistant) tutar
- `chat_turns` yalnızca `SessionStore` içinde; DB / dosya / log’a **yazılmaz**
- İstemci UI geçmişini ekranda tutabilir; sunucu persist etmez
- Loglarda soru/cevap metni **kaydedilmez** (yalnızca anonim metrik: `session_created`, `question_count`, `format`)

---

## 5. Limitler

Tüm limitler `app/core/config.py` ve `.env` üzerinden yapılandırılır.

### 5.1 Dosya yükleme

| Ayar | Env | Varsayılan | Açıklama |
|------|-----|------------|----------|
| `document_max_file_size_mb` | `DOCUMENT_MAX_FILE_SIZE_MB` | 15 | Multipart gövde üst sınırı |
| `document_allowed_formats` | — | `pdf,epub` | MIME / uzantı doğrulama |
| `document_max_pages` | `DOCUMENT_MAX_PAGES` | 40 | PDF sayfa üst sınırı |
| `document_max_chapters` | `DOCUMENT_MAX_CHAPTERS` | 80 | EPUB işlenecek bölüm üst sınırı |
| `document_max_words` | `DOCUMENT_MAX_WORDS` | 120_000 | EPUB (ve PDF) toplam kelime |
| `document_max_chars` | `DOCUMENT_MAX_CHARS` | 600_000 | Çıkarılan ham metin üst sınırı |

**PDF:** `max_pages` birincil kullanıcı limiti.  
**EPUB:** Sayfa kavramı yok; `max_chapters` + `max_words` + `max_chars` birlikte uygulanır.

Limit aşımında HTTP **413** veya **422** (tercih: `413 Payload Too Large` dosya boyutu, `422 Unprocessable Entity` içerik limiti).

### 5.2 Chunk ve embedding

| Ayar | Env | Varsayılan | Açıklama |
|------|-----|------------|----------|
| `document_chunk_size` | `DOCUMENT_CHUNK_SIZE` | 1000 | Karakter |
| `document_chunk_overlap` | `DOCUMENT_CHUNK_OVERLAP` | 200 | Karakter |
| `document_max_chunks` | `DOCUMENT_MAX_CHUNKS` | 120 | Embedding API + bellek tavanı |

Chunk üretimi `max_chunks`’a ulaşınca durur; kalan metin işlenmez (kullanıcıya `truncated: true` meta ile bildirilir).

### 5.3 Chat (soru-cevap)

| Ayar | Env | Varsayılan | Açıklama |
|------|-----|------------|----------|
| `document_max_question_length` | `DOCUMENT_MAX_QUESTION_LENGTH` | 500 | Tek soru |
| `document_max_questions_per_session` | `DOCUMENT_MAX_QUESTIONS_PER_SESSION` | 30 | Oturum başına soru |
| `document_retrieval_top_k` | `DOCUMENT_RETRIEVAL_TOP_K` | 5 | RAG’de getirilecek chunk |
| `document_max_context_chars` | `DOCUMENT_MAX_CONTEXT_CHARS` | 12_000 | LLM’e giden bağlam tavanı |
| `document_max_chat_turns_memory` | `DOCUMENT_MAX_CHAT_TURNS_MEMORY` | 4 | Ephemeral condense için tur |
| `document_chat_model` | `DOCUMENT_CHAT_MODEL` | `gemini-2.5-flash` | Cevap üretimi |
| `document_chat_temperature` | `DOCUMENT_CHAT_TEMPERATURE` | 0.2 | |
| `document_chat_max_output_tokens` | `DOCUMENT_CHAT_MAX_OUTPUT_TOKENS` | 1024 | |

### 5.4 Oturum ve eşzamanlılık

| Ayar | Env | Varsayılan | Açıklama |
|------|-----|------------|----------|
| `session_idle_ttl_minutes` | `SESSION_IDLE_TTL_MINUTES` | 45 | |
| `session_max_ttl_minutes` | `SESSION_MAX_TTL_MINUTES` | 120 | |
| `document_max_concurrent_sessions` | `DOCUMENT_MAX_CONCURRENT_SESSIONS` | 50 | Sunucu geneli (opsiyonel) |
| `document_max_sessions_per_ip_hour` | `DOCUMENT_MAX_SESSIONS_PER_IP_HOUR` | 10 | Rate limit (opsiyonel) |

### 5.5 Maliyet tahmini (örnek)

40 sayfa PDF, ~2500 karakter/sayfa → ~100k karakter → ~120 chunk (1000/200 splitter ile):

- Upload: ~120 `embed_documents` çağrısı (batch’lenebilir)
- Her soru: 1 `embed_query` + 1 chat completion

Gemini free tier embedding kotası için `embed_with_retry` ve günlük üst sınır uyarıları mevcut `gemini.py` ile uyumludur.

---

## 6. PDF ve EPUB desteği

### 6.1 PDF

**Kütüphane:** `pypdf` (PyPDF2’nin devamı; ChatPDF’deki PyPDF2 yerine)

**Çıkarma:** `PdfReader` ile sayfa sayfa `extract_text()`

**Limit kontrolü sırası:**

1. Dosya boyutu
2. MIME: `application/pdf`
3. Sayfa sayısı ≤ `document_max_pages`
4. Toplam karakter ≤ `document_max_chars`
5. Chunk sayısı ≤ `document_max_chunks`

**Metadata (chunk):** `{"page": 12, "format": "pdf"}`

**Bilinen sorunlar:** Tarama PDF, çok sütunlu layout, gömülü font — metin kalitesi düşük olabilir; kullanıcıya “metin çıkarılamadı” hatası (422).

### 6.2 EPUB

**Kütüphane:** `ebooklib` + isteğe bağlı `beautifulsoup4` (HTML temizliği)

**Çıkarma:**

1. `epub.read_epub()` ile aç
2. `ITEM_DOCUMENT` tipindeki spine öğelerini sırayla oku
3. HTML → düz metin (`get_body_content` / BeautifulSoup `get_text`)
4. Bölüm başına `chapter_title` metadata

**Limit kontrolü sırası:**

1. Dosya boyutu
2. MIME: `application/epub+zip` veya `.epub` uzantısı
3. İşlenen bölüm ≤ `document_max_chapters`
4. Kelime sayısı ≤ `document_max_words`
5. Karakter ≤ `document_max_chars`
6. Chunk ≤ `document_max_chunks`

**Metadata (chunk):** `{"chapter_index": 3, "chapter_title": "...", "format": "epub"}`

**Desteklenmeyen:** DRM, `.azw`, `.mobi` (bu aşamada reddedilir — 415 Unsupported Media Type).

### 6.3 Ortak chunk pipeline

```
extract_text(format) → normalize whitespace → RecursiveCharacterTextSplitter
  → truncate to max_chunks → GeminiEmbeddingClient.embed_documents()
  → DocumentSession
```

**Splitter:** `langchain_text_splitters.RecursiveCharacterTextSplitter`  
(ChatPDF’deki `CharacterTextSplitter` yerine; paragraf/cümle sınırlarına daha iyi uyum)

---

## 7. HTTP API sözleşmesi

Tüm endpoint’ler `Authorization: Bearer <API_KEY>` ile korunur (`settings.api_key` boşsa dev modda açık).

Prefix: `/api/v1`

### 7.1 Oturum oluştur

```
POST /api/v1/sessions
Content-Type: multipart/form-data
```

| Alan | Tip | Zorunlu |
|------|-----|---------|
| `file` | binary | Evet |

**Başarı (201):**

```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "format": "pdf",
  "filename": "kitap.pdf",
  "expiresAt": "2026-07-07T21:30:00Z",
  "status": "processing",
  "pageCount": null,
  "chapterCount": null,
  "wordCount": 0,
  "chunkCount": 0,
  "truncated": false,
  "limits": {
    "maxQuestionsRemaining": 10
  }
}
```

İstemci `GET /sessions/{id}` ile `status: "ready"` olana kadar poll eder. Embedding arka planda paralel batch’lerle (`document_embedding_concurrency`, varsayılan 15) çalışır.

**Hatalar:**

| Kod | Durum |
|-----|--------|
| 400 | Dosya yok / boş |
| 401 | API key |
| 413 | Dosya çok büyük |
| 415 | Desteklenmeyen format |
| 422 | Sayfa/bölüm/kelime/chunk limiti, metin çıkarılamadı |
| 503 | Gemini yapılandırılmamış veya embedding hatası |

### 7.2 Soru sor

```
POST /api/v1/sessions/{sessionId}/chat
Content-Type: application/json
```

```json
{
  "question": "Ana karakterin motivasyonu nedir?"
}
```

**Başarı (200):**

```json
{
  "answer": "...",
  "sources": [
    {
      "chunkIndex": 12,
      "excerpt": "İlk iki cümlelik kısa alıntı...",
      "metadata": { "page": 45, "format": "pdf" }
    }
  ],
  "sessionExpiresAt": "2026-07-07T21:35:00Z",
  "questionsRemaining": 29
}
```

**Hatalar:**

| Kod | Durum |
|-----|--------|
| 404 | Oturum yok veya süresi dolmuş |
| 429 | `max_questions_per_session` aşıldı |
| 422 | Soru boş veya çok uzun |

### 7.3 Oturum durumu (opsiyonel)

```
GET /api/v1/sessions/{sessionId}
```

```json
{
  "sessionId": "...",
  "format": "epub",
  "expiresAt": "...",
  "chunkCount": 87,
  "questionCount": 5,
  "questionsRemaining": 5,
  "status": "ready",
  "errorMessage": null,
  "chunksEmbedded": 87,
  "chunksTotal": 87,
  "pageCount": null,
  "chapterCount": 12,
  "wordCount": 42000,
  "truncated": false
}
```

`status` değerleri: `processing` | `ready` | `failed`. Chat yalnızca `ready` iken kabul edilir (`processing` → 409).

Chat mesajları **dönülmez** (geçmiş API’den okunmaz).

### 7.4 Oturumu sonlandır

```
DELETE /api/v1/sessions/{sessionId}
```

**204** — oturum ve tüm ephemeral veri silinir.

---

## 8. RAG ve sohbet mantığı

### 8.1 Condense question (takip soruları)

ChatPDF’den uyarlanan prompt; yalnızca oturumdaki son turlar kullanılır:

```
Given the following conversation and a follow up question, rephrase the follow up
question to be a standalone question, in its original language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:
```

`chat_history` boşsa soru olduğu gibi kullanılır.

### 8.2 Retrieval

1. Standalone soruyu `embed_query` ile vektörleştir
2. Oturum chunk’ları üzerinde cosine similarity (numpy veya saf Python)
3. En iyi `document_retrieval_top_k` chunk seç
4. Toplam bağlam `document_max_context_chars` ile kırp

### 8.3 Cevap üretimi (system prompt özeti)

```
You answer questions using ONLY the provided document excerpts.
- If the answer is not in the context, say you don't know.
- Do not invent facts.
- Keep direct quotes under 2 sentences; prefer paraphrase.
- Respond in the same language as the user's question.
```

Model: `ChatGoogleGenerativeAI` (`document_chat_model`).

### 8.4 Telif dostu çıktı

- `sources[].excerpt` en fazla ~200 karakter
- System prompt uzun alıntıyı sınırlar
- Cevaplar loglanmaz

---

## 9. Repo dosya yapısı (eklenecekler)

```
app/
├── api/
│   └── routers/
│       ├── semantic.py              # mevcut
│       └── sessions.py                # YENİ — upload, chat, status, delete
├── core/
│   └── config.py                      # GÜNCELLE — document_* ve session_* ayarları
├── data/
│   ├── datasources/
│   │   └── gemini.py                  # mevcut (paylaşılır)
│   └── session_store/
│       ├── __init__.py
│       ├── base.py                    # SessionStore protocol
│       ├── memory.py                  # InMemorySessionStore
│       └── redis.py                   # opsiyonel, faz 2
├── domain/
│   ├── document_chat_service.py       # YENİ — RAG orchestration
│   ├── document_chunker.py            # YENİ — splitter + limitler
│   ├── document_extractors/
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py
│   │   └── epub_extractor.py
│   └── search_service.py              # mevcut, dokunulmaz
├── models/
│   ├── schemas.py                     # GÜNCELLE — session/chat DTO
│   └── document_session.py            # YENİ — domain modelleri
└── main.py                            # GÜNCELLE — sessions router include

docs/
└── DOCUMENT_CHAT_INTEGRATION.md       # bu dosya

requirements.txt                       # GÜNCELLE — pypdf, ebooklib, langchain-text-splitters
```

**Supabase migration:** Bu özellik için **gerekmez**.

---

## 10. `config.py` ek alanları (referans)

```python
# Document chat — limits
document_max_file_size_mb: int = 15
document_max_pages: int = 40
document_max_chapters: int = 80
document_max_words: int = 120_000
document_max_chars: int = 600_000
document_chunk_size: int = 1000
document_chunk_overlap: int = 200
document_max_chunks: int = 120
document_max_question_length: int = 500
document_max_questions_per_session: int = 30
document_retrieval_top_k: int = 5
document_max_context_chars: int = 12_000
document_max_chat_turns_memory: int = 4
document_chat_model: str = "gemini-2.5-flash"
document_chat_temperature: float = 0.2
document_chat_max_output_tokens: int = 1024
document_embedding_concurrency: int = 15
document_embedding_batch_size: int = 50

# Session TTL
session_idle_ttl_minutes: int = 45
session_max_ttl_minutes: int = 120
session_cleanup_interval_seconds: int = 60

# Opsiyonel rate limits
document_max_concurrent_sessions: int = 50
document_max_sessions_per_ip_hour: int = 10

# Redis (production)
redis_url: str = ""
session_store_backend: Literal["memory", "redis"] = "memory"
```

---

## 11. Bağımlılıklar

`requirements.txt` **eklemeleri:**

```
pypdf>=4.0.0
ebooklib>=0.18
beautifulsoup4>=4.12.0
langchain-text-splitters>=0.3.0
```

**Opsiyonel (Redis store):**

```
redis>=5.0.0
```

**Eklenmeyecek:** `streamlit`, `faiss-cpu`, `sentence-transformers`, `torch`, `openai`, `PyPDF2`.

---

## 12. `main.py` entegrasyonu

```python
from app.api.routers.sessions import router as sessions_router

app.include_router(semantic_router)
app.include_router(sessions_router)
```

Uygulama başlarken (in-memory store):

```python
@app.on_event("startup")
async def start_session_cleanup():
    session_store.start_cleanup_task()
```

---

## 13. Flutter istemci entegrasyonu

### Yeni env

```json
"SEMANTIC_API_BASE_URL": "http://localhost:8000",
"SEMANTIC_API_KEY": "your-api-key"
```

(Mevcut semantic URL/key aynı servisi kullanır.)

### Önerilen UI akışı

1. Kullanıcı PDF/EPUB seçer (file picker)
2. `POST /sessions` → `sessionId` alınır; progress gösterilir
3. Chat ekranı: mesajlar **yalnızca client state**’te
4. Her mesaj: `POST /sessions/{sessionId}/chat`
5. Oturum süresi dolunca: “Yeniden yükle” mesajı
6. Çıkışta isteğe bağlı `DELETE /sessions/{sessionId}`

### Yeni Dart katmanı (öneri)

```
lib/features/document_chat/
├── data/
│   └── document_chat_api_datasource.dart
├── domain/
│   └── document_chat_repository.dart
└── presentation/
    └── document_chat_screen.dart
```

**İstemcide saklanmayan:** sunucu tarafı chat logu yok; sayfa yenilenince UI geçmişi kaybolabilir (ürün metninde belirt).

---

## 14. Güvenlik ve uyumluluk

| Konu | Uygulama |
|------|----------|
| Auth | Mevcut Bearer API key |
| Dosya doğrulama | MIME + magic bytes + uzantı |
| Boyut | `document_max_file_size_mb` + reverse proxy limiti |
| İçerik | Sayfa/bölüm/kelime/chunk tavanları |
| Gizlilik | Soru/cevap loglanmaz |
| Telif | Kullanım şartları: yalnızca hukuken kullanılabilen dosyalar; sunucu kalıcı kopya tutmaz |
| Abuse | IP başına oturum limiti, eşzamanlı oturum tavanı |

---

## 15. Hata kodları özeti

| HTTP | `detail` örneği |
|------|-----------------|
| 400 | `No file provided` |
| 401 | `Invalid or missing API key` |
| 404 | `Session not found or expired` |
| 409 | `Session is still processing...` |
| 413 | `File exceeds 20 MB limit` |
| 415 | `Unsupported format; use PDF or EPUB` |
| 422 | `PDF exceeds 40 page limit` |
| 422 | `Could not extract text from document` |
| 429 | `Question limit reached for this session` |
| 503 | `Embedding service is not configured` |

---

## 16. Uygulama fazları

### Faz 1 — MVP (tek instance)

- [ ] `config` limit alanları
- [ ] PDF extractor + chunker
- [ ] `InMemorySessionStore` + TTL cleanup
- [ ] `DocumentChatService` (RAG + Gemini)
- [ ] `sessions` router (POST upload, POST chat, DELETE)
- [ ] `requirements.txt` güncelleme
- [ ] Manuel test: küçük PDF, 3–5 soru, TTL sonrası 404

### Faz 2 — EPUB + sağlamlaştırma

- [ ] EPUB extractor
- [ ] `truncated` meta ve limit mesajları
- [ ] `GET /sessions/{id}` status
- [ ] Birim testler: extractor, chunker, cosine retrieval

### Faz 3 — Production

- [ ] `RedisSessionStore`
- [ ] IP rate limit
- [ ] Flutter `document_chat` feature
- [ ] Deploy notları (bellek tahmini: ~50 oturum × 120 chunk × 768 float)

---

## 17. Test planı

| Senaryo | Beklenen |
|---------|----------|
| 5 sayfa PDF upload | 201, `chunkCount > 0` |
| 50 sayfa PDF | 422 sayfa limiti |
| DRM’siz EPUB | 201, `format: epub` |
| Geçersiz .docx | 415 |
| 31. soru | 429 |
| Süresi dolmuş session ile chat | 404 |
| DELETE sonrası chat | 404 |
| Takip sorusu (“peki ya o?”) | Anlamlı cevap (condense çalışır) |
| İlgisiz soru | “Bilmiyorum” tarzı cevap |

---

## 18. Deploy notları

- **Bellek:** Her oturum ~120 chunk × (metin + 768×4 byte vektör) ≈ birkaç MB; `document_max_concurrent_sessions` ile sınırla
- **CPU:** Embedding upload sırasında yoğun; uzun işlem için client timeout ≥ 120s
- **Ölçek:** Horizontal scale → Redis session store zorunlu
- **Mevcut semantic search:** Aynı `GOOGLE_API_KEY`; kota paylaşılır — limitleri izle

---

## 19. Referanslar

- ChatPDF kaynak: https://github.com/ArmaanSeth/ChatPDF
- Mevcut mimari: [ARCHITECTURE.md](../ARCHITECTURE.md)
- Gemini embedding client: `app/data/datasources/gemini.py`
- Semantic router örneği: `app/api/routers/semantic.py`

---

*Son güncelleme: entegrasyon spesifikasyonu — uygulama henüz kodlanmadı; bu doküman implementasyon rehberidir.*
