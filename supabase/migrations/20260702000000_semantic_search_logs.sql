-- Semantic search analytics (separate from keyword search_logs).

create table if not exists public.semantic_search_logs (
  id            uuid primary key default gen_random_uuid(),
  user_id       uuid references auth.users (id) on delete set null,
  query         text not null,
  mode          text not null check (mode in ('simple', 'advanced')),
  category      text,
  tone          text,
  result_count  integer not null default 0,
  created_at    timestamptz not null default now()
);

create index if not exists semantic_search_logs_created_at_idx
  on public.semantic_search_logs using btree (created_at desc);

create index if not exists semantic_search_logs_mode_idx
  on public.semantic_search_logs (mode);

alter table public.semantic_search_logs enable row level security;

create policy "semantic_search_logs_insert_own"
  on public.semantic_search_logs for insert
  to authenticated, anon
  with check (user_id is null or user_id = auth.uid());

create policy "semantic_search_logs_select_own"
  on public.semantic_search_logs for select
  to authenticated
  using (user_id = auth.uid());
