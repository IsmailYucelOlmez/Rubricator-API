-- pgvector semantic search RPC (called by FastAPI with service_role).

create or replace function public.semantic_search_books(
  p_query_embedding extensions.vector(768),
  p_category        text default null,
  p_limit           int  default 16,
  p_initial_k       int  default 50
)
returns table (
  isbn13          text,
  title           text,
  authors         text,
  description     text,
  thumbnail_url   text,
  simple_category text,
  emotion_scores  jsonb,
  similarity      float
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
    1 - (bc.embedding <=> p_query_embedding) as similarity
  from public.book_catalog bc
  where (p_category is null or p_category = 'All' or bc.simple_category = p_category)
  order by bc.embedding <=> p_query_embedding
  limit least(greatest(p_limit, 1), p_initial_k);
$$;

grant execute on function public.semantic_search_books(extensions.vector, text, int, int)
  to service_role;
