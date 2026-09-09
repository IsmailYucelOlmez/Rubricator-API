-- Add language metadata to book_catalog and thread it through semantic search
-- so language="tr" searches never fall back to the Google Books API path.
-- Existing rows (books_with_emotions.csv, English) default to 'en'.

alter table public.book_catalog
  add column if not exists language text not null default 'en';

create index if not exists book_catalog_language_idx
  on public.book_catalog (language);

drop function if exists public.semantic_search_books(extensions.vector, text, int, int);

create or replace function public.semantic_search_books(
  p_query_embedding extensions.vector(768),
  p_category        text default null,
  p_limit           int  default 16,
  p_initial_k       int  default 50,
  p_language        text default null
)
returns table (
  isbn13          text,
  title           text,
  authors         text,
  description     text,
  thumbnail_url   text,
  simple_category text,
  emotion_scores  jsonb,
  similarity      float,
  source          text,
  language        text
)
language sql
stable
security definer
set search_path = public, extensions
as $$
  select
    bc.isbn13,
    bc.title,
    bc.authors,
    bc.description,
    bc.thumbnail_url,
    bc.simple_category,
    bc.emotion_scores,
    1 - (bc.embedding <=> p_query_embedding) as similarity,
    bc.source,
    bc.language
  from public.book_catalog bc
  where (p_category is null or p_category = 'All' or bc.simple_category = p_category)
    and (p_language is null or bc.language = p_language)
  order by bc.embedding <=> p_query_embedding
  limit least(greatest(p_limit, 1), p_initial_k);
$$;

grant execute on function public.semantic_search_books
  (extensions.vector, text, int, int, text) to service_role;
