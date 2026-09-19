-- Relevance feedback on semantic search results: a signed-in user marks a result as
-- relevant (+1) or irrelevant (-1) *for the query they typed*.
--
--   * Flutter writes through submit_semantic_feedback(); identity is auth.uid(),
--     never a client-supplied value. get_my_semantic_feedback() restores button state.
--   * FastAPI (service_role) only reads aggregates through get_semantic_feedback().
--   * Both tables have RLS enabled and no policies: no direct client access.
--
-- Flutter usage (supabase.rpc):
--   submit_semantic_feedback(p_query, p_isbn13, p_vote, [p_language, p_result_position,
--                            p_similarity, p_mode, p_category, p_tone])
--     p_vote: 1 = relevant, -1 = irrelevant, 0 = remove my vote
--     returns 'new' | 'changed' | 'unchanged' | 'removed'
--   get_my_semantic_feedback(p_query, [p_language]) -> rows (isbn13, vote)
-- Pass the query exactly as the user typed it (and the same p_language as the search).

create table if not exists public.semantic_feedback (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid not null references auth.users (id) on delete cascade,
  query_key          text not null,
  query_text         text not null,
  isbn13             text not null,
  vote               smallint not null check (vote in (-1, 1)),
  result_position    integer check (result_position is null or result_position >= 0),
  similarity_at_vote real,
  mode               text,
  category           text,
  tone               text,
  language           text,
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now(),
  unique (user_id, query_key, isbn13)
);

create index if not exists semantic_feedback_user_recent_idx
  on public.semantic_feedback (user_id, updated_at desc);

-- Per (query, book) vote counters; maintained only by the trigger below.
create table if not exists public.semantic_feedback_stats (
  query_key  text not null,
  isbn13     text not null,
  up         integer not null default 0 check (up >= 0),
  down       integer not null default 0 check (down >= 0),
  updated_at timestamptz not null default now(),
  primary key (query_key, isbn13)
);

alter table public.semantic_feedback enable row level security;
alter table public.semantic_feedback_stats enable row level security;

-- Deterministic key for "the same query": NFKC, lower case, collapsed whitespace,
-- trailing punctuation dropped, plus the language. All readers and writers go through
-- this one function so keys can never disagree between the app and the API.
create or replace function public.semantic_query_key(
  p_query    text,
  p_language text default null
)
returns text
language sql
immutable
set search_path = public
as $$
  select encode(
    sha256(convert_to(
      regexp_replace(
        lower(regexp_replace(btrim(normalize(left(coalesce(p_query, ''), 500), NFKC)), '\s+', ' ', 'g')),
        '[[:punct:][:space:]]+$',
        ''
      )
      || '|' || coalesce(nullif(btrim(lower(p_language)), ''), ''),
      'UTF8'
    )),
    'hex'
  );
$$;

-- Keeps semantic_feedback_stats in sync for inserts, vote flips, removals and
-- cascade deletes (deleting an account also withdraws its votes from the counters).
create or replace function public.semantic_feedback_apply_stats()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_key  text;
  v_isbn text;
  v_up   integer := 0;
  v_down integer := 0;
begin
  if tg_op = 'DELETE' then
    v_key := old.query_key;
    v_isbn := old.isbn13;
  else
    v_key := new.query_key;
    v_isbn := new.isbn13;
  end if;

  if tg_op in ('INSERT', 'UPDATE') then
    if new.vote = 1 then v_up := v_up + 1; else v_down := v_down + 1; end if;
  end if;
  if tg_op in ('UPDATE', 'DELETE') then
    if old.vote = 1 then v_up := v_up - 1; else v_down := v_down - 1; end if;
  end if;

  insert into public.semantic_feedback_stats as s (query_key, isbn13, up, down)
  values (v_key, v_isbn, greatest(v_up, 0), greatest(v_down, 0))
  on conflict (query_key, isbn13) do update
    set up = greatest(s.up + v_up, 0),
        down = greatest(s.down + v_down, 0),
        updated_at = now();

  delete from public.semantic_feedback_stats
  where query_key = v_key and isbn13 = v_isbn and up = 0 and down = 0;

  return null;
end;
$$;

drop trigger if exists semantic_feedback_stats_trg on public.semantic_feedback;
create trigger semantic_feedback_stats_trg
  after insert or update of vote or delete on public.semantic_feedback
  for each row execute function public.semantic_feedback_apply_stats();

create or replace function public.submit_semantic_feedback(
  p_query           text,
  p_isbn13          text,
  p_vote            smallint,
  p_language        text default null,
  p_result_position integer default null,
  p_similarity      real default null,
  p_mode            text default null,
  p_category        text default null,
  p_tone            text default null
)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user     uuid := auth.uid();
  v_query    text := left(btrim(coalesce(p_query, '')), 500);
  v_isbn     text := upper(coalesce(p_isbn13, ''));
  v_language text := nullif(left(btrim(lower(coalesce(p_language, ''))), 16), '');
  v_key      text;
  v_inserted boolean;
begin
  if v_user is null then
    raise exception 'authentication required' using errcode = '28000';
  end if;
  if p_vote is null or p_vote not in (-1, 0, 1) then
    raise exception 'vote must be -1, 0 or 1' using errcode = '22023';
  end if;
  if v_query = '' then
    raise exception 'query is required' using errcode = '22023';
  end if;
  if v_isbn !~ '^[0-9]{9,12}[0-9X]$' then
    raise exception 'invalid isbn13' using errcode = '22023';
  end if;

  v_key := public.semantic_query_key(p_query, p_language);

  if p_vote = 0 then
    delete from public.semantic_feedback
    where user_id = v_user and query_key = v_key and isbn13 = v_isbn;
    return case when found then 'removed' else 'unchanged' end;
  end if;

  -- Cheap abuse brake: one account can touch at most 200 distinct votes per hour.
  if (
    select count(*) from public.semantic_feedback
    where user_id = v_user and updated_at > now() - interval '1 hour'
  ) >= 200 then
    raise exception 'feedback rate limit exceeded' using errcode = 'PT429';
  end if;

  insert into public.semantic_feedback as f (
    user_id, query_key, query_text, isbn13, vote,
    result_position, similarity_at_vote, mode, category, tone, language
  )
  values (
    v_user, v_key, v_query, v_isbn, p_vote,
    p_result_position, p_similarity, left(p_mode, 20), left(p_category, 50), left(p_tone, 50), v_language
  )
  on conflict (user_id, query_key, isbn13) do update
    set vote = excluded.vote,
        result_position = excluded.result_position,
        similarity_at_vote = excluded.similarity_at_vote,
        mode = excluded.mode,
        category = excluded.category,
        tone = excluded.tone,
        updated_at = now()
    where f.vote <> excluded.vote
  returning (xmax = 0) into v_inserted;

  if not found then
    return 'unchanged';
  end if;
  return case when v_inserted then 'new' else 'changed' end;
end;
$$;

create or replace function public.get_my_semantic_feedback(
  p_query    text,
  p_language text default null
)
returns table (isbn13 text, vote smallint)
language sql
stable
security definer
set search_path = public
as $$
  select f.isbn13, f.vote
  from public.semantic_feedback f
  where f.user_id = auth.uid()
    and f.query_key = public.semantic_query_key(p_query, p_language);
$$;

create or replace function public.get_semantic_feedback(
  p_query    text,
  p_language text default null
)
returns table (isbn13 text, up integer, down integer)
language sql
stable
security definer
set search_path = public
as $$
  select s.isbn13, s.up, s.down
  from public.semantic_feedback_stats s
  where s.query_key = public.semantic_query_key(p_query, p_language)
    and s.up + s.down > 0;
$$;

-- Supabase grants EXECUTE on new functions to anon/authenticated by default; lock down.
revoke all on function public.semantic_query_key(text, text) from public, anon, authenticated;
revoke all on function public.semantic_feedback_apply_stats() from public, anon, authenticated;
revoke all on function public.submit_semantic_feedback(text, text, smallint, text, integer, real, text, text, text)
  from public, anon, authenticated;
revoke all on function public.get_my_semantic_feedback(text, text) from public, anon, authenticated;
revoke all on function public.get_semantic_feedback(text, text) from public, anon, authenticated;

grant execute on function public.submit_semantic_feedback(text, text, smallint, text, integer, real, text, text, text)
  to authenticated;
grant execute on function public.get_my_semantic_feedback(text, text) to authenticated;
grant execute on function public.get_semantic_feedback(text, text) to service_role;
