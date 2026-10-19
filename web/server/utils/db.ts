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
`;

