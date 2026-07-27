-- Semantic book catalog with pgvector embeddings (Gemini embedding-001, 768 dims).

create extension if not exists vector with schema extensions;

create table if not exists public.book_catalog (
  isbn13            text primary key,
  isbn10            text,
  title             text not null,
  authors           text not null,
  description       text not null default '',
  thumbnail_url     text,
  google_volume_id  text,
  simple_category   text,
  published_year    integer,
  emotion_scores    jsonb not null default '{}'::jsonb,
  embedding         extensions.vector(768) not null,
  source            text not null default 'local',
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

create index if not exists book_catalog_embedding_idx
  on public.book_catalog
  using ivfflat (embedding extensions.vector_cosine_ops)
  with (lists = 100);

create index if not exists book_catalog_category_idx
  on public.book_catalog (simple_category);

alter table public.book_catalog enable row level security;

create policy "book_catalog_select_all"
  on public.book_catalog for select
  to anon, authenticated
  using (true);
