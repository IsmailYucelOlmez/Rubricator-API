-- ISBN → Google Books volume ID cache for semantic discovery → detail navigation.

create table if not exists public.book_identity_cache (
  isbn13            text primary key,
  google_volume_id  text not null,
  resolved_title    text,
  resolved_at       timestamptz not null default now(),
  resolve_method    text not null default 'isbn'
);

alter table public.book_identity_cache enable row level security;

create policy "book_identity_cache_select_all"
  on public.book_identity_cache for select
  to anon, authenticated
  using (true);

create or replace function public.upsert_book_identity_cache(
  p_isbn13           text,
  p_google_volume_id text,
  p_resolved_title   text default null,
  p_resolve_method   text default 'isbn'
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.book_identity_cache (
    isbn13,
    google_volume_id,
    resolved_title,
    resolve_method,
    resolved_at
  )
  values (
    p_isbn13,
    p_google_volume_id,
    p_resolved_title,
    coalesce(p_resolve_method, 'isbn'),
    now()
  )
  on conflict (isbn13) do update set
    google_volume_id = excluded.google_volume_id,
    resolved_title = coalesce(excluded.resolved_title, book_identity_cache.resolved_title),
    resolve_method = excluded.resolve_method,
    resolved_at = now();
end;
$$;

grant execute on function public.upsert_book_identity_cache(text, text, text, text)
  to authenticated, anon;
