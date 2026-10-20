// PostgreSQL для аккаунтов, облачной полки и отзывов. Адрес — NUXT_DATABASE_URL; без него
// сканер и локальная полка работают как раньше, а ручки аккаунта честно отвечают 503.
import postgres from "postgres";

let client: postgres.Sql | null = null;
let migrated: Promise<unknown> | null = null;

export function dbEnabled() {
  return Boolean(useRuntimeConfig().databaseUrl);
}

// Схема применяется при первом обращении к базе, а не при старте: так сервер поднимается и без
// живой базы, а сбой подключения повторяется на следующем запросе.
export async function useDb() {
  if (!dbEnabled()) throw createError({ statusCode: 503, statusMessage: "Accounts are not configured" });
  client ??= postgres(useRuntimeConfig().databaseUrl, { max: 10, idle_timeout: 30, onnotice: () => {} });
  migrated ??= client.unsafe(SCHEMA).catch((error) => {
    migrated = null;
    console.error("PostgreSQL is unavailable:", (error as Error).message);
    throw createError({ statusCode: 503, statusMessage: "Database is unavailable" });
  });
  await migrated;
  return client;
}

// Схема идемпотентна: таблиц немного, отдельный инструмент миграций здесь не окупается.
// Новые поля добавлять через ADD COLUMN IF NOT EXISTS.
export const SCHEMA = `
create table if not exists users (
  id bigserial primary key,
  phone text not null unique,
  name text not null default '',
  email text,
  avatar bytea,
  avatar_type text,
  avatar_version integer not null default 0,
  adult_confirmed_at timestamptz,
  default_share boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create table if not exists auth_codes (
  phone text primary key,
  code_hash text not null,
  attempts integer not null default 0,
  sent_at timestamptz not null default now(),
  expires_at timestamptz not null
);
create table if not exists sessions (
  token_hash text primary key,
  user_id bigint not null references users(id) on delete cascade,
  created_at timestamptz not null default now(),
  expires_at timestamptz not null
);
create index if not exists sessions_user on sessions (user_id);
create table if not exists shelf_entries (
  user_id bigint not null references users(id) on delete cascade,
  slug text not null,
  card jsonb not null,
  analogues jsonb not null default '[]',
  status text not null,
  reaction text,
  dishes text[] not null default '{}',
  note text not null default '',
  remind_at bigint,
  added_at bigint not null,
  updated_at bigint not null,
  deleted boolean not null default false,
  primary key (user_id, slug)
);
create index if not exists shelf_entries_slug on shelf_entries (slug) where not deleted;
create table if not exists reviews (
  user_id bigint not null references users(id) on delete cascade,
  slug text not null,
  liked boolean not null,
  dishes text[] not null default '{}',
  note text not null default '',
  anonymous boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, slug)
);
create index if not exists reviews_slug on reviews (slug, updated_at desc);
-- Игра «Моя винодельня»: состояние игрока целиком в jsonb (правила — utils/game.mjs), журнал
-- действий для идемпотентности и аудита, журнал наград за действия в сервисе и Винордл.
create table if not exists game_profiles (
  user_id bigint primary key references users(id) on delete cascade,
  state jsonb not null,
  version integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
create table if not exists game_ledger (
  id bigserial primary key,
  user_id bigint not null references users(id) on delete cascade,
  request_id text,
  action text not null,
  delta jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create unique index if not exists game_ledger_request on game_ledger (user_id, request_id) where request_id is not null;
create table if not exists game_service_rewards (
  user_id bigint not null references users(id) on delete cascade,
  kind text not null,
  ref text not null,
  day date not null,
  reward jsonb not null,
  seen boolean not null default false,
  created_at timestamptz not null default now(),
  primary key (user_id, kind, ref)
);
create index if not exists game_service_rewards_day on game_service_rewards (user_id, kind, day);
create table if not exists wordle_attempts (
  user_id bigint not null references users(id) on delete cascade,
  day date not null,
  guesses integer[] not null default '{}',
  solved boolean not null default false,
  finished boolean not null default false,
  updated_at timestamptz not null default now(),
  primary key (user_id, day)
);
create index if not exists wordle_attempts_day on wordle_attempts (day);
-- Игровые пуши: подписки устройств игрока и журнал отправленного (utils/game-push.mjs).
create table if not exists game_push (
  user_id bigint primary key references users(id) on delete cascade,
  devices jsonb not null default '[]',
  log jsonb not null default '{}',
  updated_at timestamptz not null default now()
);
`;

