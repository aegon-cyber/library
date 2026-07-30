-- ============================================================
-- library-agent Supabase 初始化 SQL
-- 请在 Supabase Dashboard → SQL Editor 中粘贴执行
-- ============================================================

-- 1. 启用 pgvector 扩展
create extension if not exists vector;

-- 2. 建 images 表
create table if not exists images (
    id            serial primary key,
    file_name     text not null,
    file_path     text not null,
    thumbnail_path text,
    uploader      text not null default 'test_user',
    upload_time   timestamptz not null default now(),
    description   text,
    extra_description text,
    embedding     vector(2048)          -- ZhipuAI Embedding-3 输出 1024 维
);

-- 3. 索引：IVFFlat 加速向量搜索（数据量 >1000 后效果显著）
create index if not exists idx_images_embedding
    on images using ivfflat (embedding vector_cosine_ops)
    with (lists = 50);

-- 4. 结构化过滤索引
create index if not exists idx_images_uploader
    on images (uploader);

create index if not exists idx_images_upload_time
    on images (upload_time);

-- 5. 相似度搜索函数
create or replace function match_images(
    query_embedding  vector(2048),
    match_threshold  float default 0.0,
    match_count      int   default 10,
    filter_uploader  text  default null,
    filter_date_from text  default null,
    filter_date_to   text  default null
)
returns table (
    id               int,
    file_name        text,
    file_path        text,
    thumbnail_path   text,
    uploader         text,
    upload_time      timestamptz,
    description      text,
    extra_description text,
    similarity       float
)
language plpgsql
as $$
begin
    return query
    select
        i.id,
        i.file_name,
        i.file_path,
        i.thumbnail_path,
        i.uploader,
        i.upload_time,
        i.description,
        i.extra_description,
        1 - (i.embedding <=> query_embedding) as similarity
    from images i
    where
        case
            when filter_uploader is not null then i.uploader = filter_uploader
            else true
        end
        and case
            when filter_date_from is not null then i.upload_time >= filter_date_from::timestamptz
            else true
        end
        and case
            when filter_date_to is not null then i.upload_time <= filter_date_to::timestamptz
            else true
        end
        and 1 - (i.embedding <=> query_embedding) > match_threshold
    order by i.embedding <=> query_embedding
    limit match_count;
end;
$$;
