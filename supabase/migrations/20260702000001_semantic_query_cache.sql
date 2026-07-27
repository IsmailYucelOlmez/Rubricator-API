-- Query result cache for advanced semantic search (FastAPI service_role writes).

create table if not exists public.semantic_query_cache (
  cache_key     text primary key,
  isbn13_list   text[] not null default '{}',
  rewrite_json  jsonb,
  expires_at    timestamptz not null,
  created_at    timestamptz not null default now()
);

create index if not exists semantic_query_cache_expires_at_idx
  on public.semantic_query_cache (expires_at);

alter table public.semantic_query_cache enable row level security;

-- No client policies: reads/writes via service_role only.
