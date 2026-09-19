# RAG ve Arama Güncellemesi: Genel Rapor

Tarih: 2026-09-19 · Kapsam: `19c75d4..HEAD` (10 commit, 42 dosya, +3505 / −24) · Test: 0 → 219

## 1. Özet

Kod tabanındaki AI/RAG yapıları incelendi ve şu eksikler kapatıldı:

| Alan | Sorun | Yapılan |
|---|---|---|
| Güvenlik | `API_KEY` boşsa kimlik doğrulama sessizce kapanıyordu | Production'da `API_KEY` yoksa uygulama başlamaz |
| Güvenlik | Yüklenen belge ve dosya adı filtresiz LLM'e giriyordu | Sınırlayıcı + kaçış, dosya adı temizleme, şüpheli kalıp loglama |
| Kota | Oturumlar kota sınırına birlikte çarpıp ayrı ayrı bekliyordu | Süreç geneli limiter + ortak cooldown |
| Doküman chat | Skor ne olursa olsun 5 chunk gidiyordu, tekrarlar üst sıraları dolduruyordu | Alaka eşiği, MMR, bütçeye göre doldurma, uyarlanabilir top_k |
| Katalog arama | `initial_top_k=50` fiilen kullanılmıyordu, sonuçlar birbirine çok benzeyebiliyordu | 50 aday + MMR ile çeşitlendirme |
| Geri bildirim | Kullanıcı sinyali sıralamaya hiç yansımıyordu | Alakalı/alakasız oyları: toplama, skor düzeltmesi, oturum içi sorgu iyileştirme |
| Gözlemlenebilirlik | Skor ve cache metrikleri yoktu | Yapılandırılmış loglar (sorgu metni yazılmaz) |
| Test | Hiç test yoktu | 219 test, her commit izole ve `.env`'siz ağaçta doğrulandı |

## 2. Commit'ler

| # | Commit | Konu | Test |
|---|---|---|---|
| 1 | `da93797` | pytest kurulumu + normalizer/cache-key testleri | 17 |
| 2 | `a8dc755` | Production'da API_KEY zorunluluğu (`ENVIRONMENT`) | 22 |
| 3 | `d7944ef` | Gemini embedding istekleri için süreç geneli limiter | 31 |
| 4 | `dcb2247` | Doküman chat retrieval kalitesi | 92 |
| 5 | `e75de64` | Prompt injection sertleştirme + retrieval skor logları | 117 |
| 6 | `e6ab005` | Katalog aramada MMR | 122 |
| 7 | `9a34d8f` | Arama ve query-cache metrik logları | 127 |
| 8 | `23909ce` | Geri bildirim: Supabase şeması/RPC'leri + oy okuma | 143 |
| 9 | `c4caa91` | Oyları sıralamaya uygulama (bayrak arkasında) | 173 |
| 10 | `457af9d` | Rocchio ile oturum içi sorgu iyileştirme | 219 |

## 3. Değişiklik ayrıntıları

### 3.1 Doküman chat (commit 4-5)
- **Alaka eşiği:** `DOCUMENT_RETRIEVAL_MIN_SCORE` (0.3) altındaki chunk'lar atılır. Hiçbiri geçmezse modele "ilgili alıntı bulunamadı" işareti gider, kaynak listesi boş döner.
- **MMR:** en iyi `top_k × 4` chunk'tan `top_k` seçilir (λ=0.7).
- **Bütçeli doldurma:** karakter bütçesi bütün chunk'larla doldurulur, cümle ortasından kesilmez, sığmayan chunk atlanır. Seçilenler doküman sırasına dizilir, `sources` yalnızca bağlama girenleri listeler.
- **Etiket ve zenginleştirme:** her alıntı `[Page N]` / `[Section N: başlık]` ile etiketlenir. Anlamlı EPUB başlıkları yalnızca embedding metnine eklenir. (`ch03`, `split_005` gibi dosya adı kaynaklı başlıklar filtrelenir.)
- **Uyarlanabilir top_k ve condense:** LLM çağrısı olmadan sezgisel sınıflandırma. Özet/analiz/karşılaştırma sorularında `top_k` 5 yerine 10. `condense_question` çağrısı yalnızca takip sorularında (zamir/referans, kısa soru).
- **Sistem promptu** 5 kurala sıkılaştırıldı.
- **Güvenlik:** alıntılar `<excerpts>` içinde gider, belgedeki sahte etiketler kaçırılır, dosya adı temizlenir (kullanıcının doğrudan kontrol ettiği enjeksiyon yoluydu), bilinen kalıplar yükleme sırasında uyarı olarak loglanır (engellenmez).

### 3.2 Katalog arama (commit 6, 9, 10)
- **MMR:** Qdrant'tan `initial_top_k` (50) aday, vektörleriyle çekilir; `limit`'e MMR ile indirilir. Önceki kod `min(limit, initial_k)` kullandığı için 50'lik havuz fiilen hiç oluşmuyordu.
- **Oy düzeltmesi (bayrak arkasında):** `delta = FEEDBACK_WEIGHT × (up − down) / (up + down + prior)`, çiftte en az `FEEDBACK_MIN_VOTES` oy varsa. Vektör skoruna eklenir, MMR'den önce uygulanır. Oy varsa 50 adayın tamamı çekilir, böylece 17-50. sıradaki bir kitap ilk 16'ya girebilir. Oy sorgusu embedding ile paralel başlar, en fazla 150 ms beklenir, hata/zaman aşımında oysuz devam edilir.
- **Rocchio:** `q' = birim(q) + 0.5·ort(alakalı) − 0.3·ort(alakasız)`; alakasız işaretlenenler sonuçlardan çıkarılır. Oy verisi gerektirmez, hata durumunda orijinal sorguyla devam eder.

### 3.3 Geri bildirim altyapısı (commit 8)
- `semantic_feedback` (oy olayları, `unique (user_id, query_key, isbn13)`) ve `semantic_feedback_stats` (sayaçlar, trigger ile tutulur). İkisinde RLS açık, istemci politikası yok.
- `submit_semantic_feedback`: kimlik `auth.uid()`; oy verme/çevirme/geri alma; saatte en fazla 200 oy.
- `get_my_semantic_feedback`: kullanıcının kendi oyları (buton durumu için). `get_semantic_feedback`: yalnızca `service_role`.
- Sorgu anahtarı tek SQL fonksiyonunda (`semantic_query_key`) hesaplanır, yazan ve okuyan aynı yolu kullanır.
- Hesap silinince oylar sayaçlardan da düşer.

### 3.4 Kota (commit 3)
`GlobalEmbeddingLimiter`: süreç geneli eşzamanlılık sınırı (`EMBEDDING_MAX_CONCURRENCY`=10), isteğe bağlı aralık (`EMBEDDING_REQUESTS_PER_MINUTE`, 0=kapalı) ve 429 sonrası ortak cooldown. Sorgu embedding'i kuyruğa girmez ama 429 alırsa toplu lane'i geri çeker.

## 4. Sözleşme değişiklikleri

**Arama isteği** (`POST /api/v1/semantic/search`) yeni, isteğe bağlı alan:
```json
{ "query": "...", "feedback": { "relevant": ["978..."], "irrelevant": ["978..."] } }
```
En fazla 5'er ISBN, biçim hatası 422. Alan yoksa davranış değişmez.

**Supabase RPC'leri (Flutter):**
`submit_semantic_feedback(p_query, p_isbn13, p_vote, [p_language, p_result_position, p_similarity, p_mode, p_category, p_tone])`, `p_vote`: 1 alakalı, −1 alakasız, 0 kaldır; dönüş `new | changed | unchanged | removed`. `p_query` ve `p_language` aramada gönderilenle birebir aynı olmalı, aksi halde oy başka bir sorguya yazılır.

**`similarity` alanı:** iyileştirilmiş (Rocchio) aramalarda kaydırılmış sorguya göre ölçülür. Oy düzeltmesinde ham vektör skoru olarak kalır.

## 5. Yeni ayarlar

| Değişken | Varsayılan | Not |
|---|---|---|
| `ENVIRONMENT` | `development` | `production` iken `API_KEY` zorunlu |
| `EMBEDDING_MAX_CONCURRENCY` | 10 | Süreç geneli |
| `EMBEDDING_REQUESTS_PER_MINUTE` | 0 | 0 = sınırsız; kendi Gemini katmanınıza göre ayarlayın |
| `DOCUMENT_RETRIEVAL_MIN_SCORE` | 0.3 | Kalibre edilmedi |
| `DOCUMENT_RETRIEVAL_TOP_K_COMPLEX` | 10 | |
| `DOCUMENT_MMR_LAMBDA` / `DOCUMENT_MMR_FETCH_MULTIPLIER` | 0.7 / 4 | λ=1.0 MMR'yi kapatır |
| `SEARCH_MMR_LAMBDA` | 0.7 | λ=1.0 eski davranışı geri getirir |
| `FEEDBACK_RERANK_ENABLED` | `false` | Oyları sıralamaya uygular |
| `FEEDBACK_WEIGHT` / `FEEDBACK_PRIOR_STRENGTH` / `FEEDBACK_MIN_VOTES` | 0.05 / 5 / 3 | Kalibre edilmedi |
| `FEEDBACK_CACHE_TTL_SECONDS` / `FEEDBACK_LOOKUP_TIMEOUT_SECONDS` | 60 / 0.15 | |
| `REFINE_RELEVANT_WEIGHT` / `REFINE_IRRELEVANT_WEIGHT` | 0.5 / 0.3 | |

## 6. Varsayılan olarak açık gelen davranış değişiklikleri

Bunlar bayrak olmadan yayına çıkar; devreye almadan önce bilinmeli:
- **Katalog aramada MMR açık.** Sonuçlar çeşitlenir, ilk sonuç en benzeri ama sonrası kesin benzerlik sırasında değildir. Her aramada 50 aday vektörü de taşınır (768 boyut), gecikme artabilir; ölçülmedi. Geri almak için `SEARCH_MMR_LAMBDA=1.0`.
- **Doküman chat daha sık "bulunamadı" diyebilir** (eşik), kaynaklar doküman sırasında ve yalnızca kullanılanlar.
- **Büyük embedding işlerinde süreç geneli 10 eşzamanlı istek sınırı** uygulanır (script'ler dahil).
- Oy sıralaması (`FEEDBACK_RERANK_ENABLED`) ve production kontrolü (`ENVIRONMENT`) varsayılan kapalıdır.

## 7. Devreye alma sırası

1. Kodu yayınlayın, `document_retrieval` ve `semantic_search` loglarını izleyin; MMR gecikmesi sorun olursa `SEARCH_MMR_LAMBDA=1.0`.
2. `supabase/migrations/20260919000000_semantic_feedback.sql` uygulayın. İki kullanıcıyla oy verme/çevirme/geri alma deneyip `semantic_feedback_stats` sayaçlarını kontrol edin.
3. Flutter'da "Bu sonuç aramanla alakalı mı? Alakalı / Alakasız" düğmelerini ve isteğe bağlı `feedback` alanını ekleyin.
4. Yeterli oy birikince `FEEDBACK_RERANK_ENABLED=true`; `feedback_lookup` ve `feedback_applied` loglarını izleyin.

## 8. Doğrulama durumu

**Yapıldı:** 219 birim/entegrasyon testi (sahte istemcilerle); her commit ayrı, `.env`'siz bir ağaçta çalıştırıldı; son ağaç nihai dosyalarla birebir eşleşti; OpenAPI şeması üretimi denendi.

**Yapılmadı:**
- Canlı Gemini, Qdrant ve Supabase ile hiçbir deneme yapılmadı.
- Migration gerçek bir Postgres'te **çalıştırılmadı** (Docker kapalıydı). Yalnızca `pglast` ile sözdizimi ve PL/pgSQL gövdeleri ayrıştırıldı; trigger, `on conflict … returning (xmax = 0)` ve grant davranışı canlıda doğrulanmalı.
- Gecikme ölçülmedi; alaka eşiği, MMR λ ve oy ağırlıkları gerçek veriyle ayarlanmadı.

## 9. Bilinen sınırlamalar ve riskler

- **Prompt injection çözülmedi**, yalnızca kolay yollar kapatıldı. Etki alanı dar (araç çağrısı yok, başka kullanıcı verisi yok) ama sıfır değil.
- **Limiter ve oturum deposu tek süreçlidir.** Birden fazla instance'ta her biri kendi sayacını tutar; Redis oturum deposu hâlâ uygulanmadı.
- **Oy etkisi başta seyrek olacak:** sorgular çeşitli, aynı çifte 3 oy toplanması zaman alır. Kullanıcılar çoğunlukla üst sonuçlara oy verdiği için konum yanlılığı var (`result_position` bunun analizi için kaydedilir).
- **Oyların sıralamaya yansıması gecikmeli** (instance başına 60 sn cache).
- **Supabase katalog backend'i:** vektör katalogu uzun süre devre dışı kalacağı için MMR, Rocchio ve havuz derinliği yalnızca Qdrant'ta tam çalışır. Supabase yolunda oylar yalnızca dönen satırları yeniden sıralar, Rocchio yok sayılır (uyarı loglanır). `CATALOG_BACKEND` varsayılanı hâlâ `supabase`; production ortamında env'in `qdrant` olduğundan emin olun.
- **Gizlilik:** `query_text` ile `user_id` artık birlikte saklanıyor (hesap silinince cascade). Gizlilik metni/KVKK açısından değerlendirilmeli.
- **Dokümantasyon:** `README.md` ve `ARCHITECTURE.md` bu güncellemeyi yansıtacak şekilde güncellenmedi.

## 10. Bu güncellemenin dışında kalanlar

- Google Books'tan gelen kitaplar için duygu skorları (HuggingFace modeli): dinamik eklenen kitaplarda `emotion_scores` hâlâ boş, ton sıralaması onlar için çalışmıyor.
- Redis oturum deposu, LangSmith/tracing, GraphRAG/agentic RAG (ürün ölçeğine uygun bulunmadı), bağlam sıkıştırma (bağlam penceresi darboğaz değil).
- Arama log insert'ünün arka plana alınması (`semantic.py`'de yanıtı hâlâ bekletiyor).
- Yakın sorgulara oy yayma ve oy ağırlığının zamanla azalması.

## 11. Log alanları (izleme için)

| Log | Alanlar |
|---|---|
| `document_retrieval` | complexity, top_k, chunks, relevant, returned, selected, best_score, scores, min_score, condensed, no_match |
| `prompt_injection_signals` | flagged_chunks, signals |
| `semantic_search` | mode, language, category, tone, rewritten, results, top/lowest_similarity |
| `advanced_search` | cache=hit/miss, rewrite, api_queries, volumes, new_candidates, upserted |
| `query_cache` | result=hit/miss/expired, hits, misses, hit_rate |
| `feedback_lookup` | ms, rows (veya zaman aşımı/hata uyarısı) |
| `feedback_applied` | adjusted_books, max_abs_delta |
| `query_refined` | relevant=x/y, irrelevant=x/y, excluded |

Hiçbir log satırı kullanıcı sorgusunu veya soru metnini içermez.
