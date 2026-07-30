-- 修复：先删掉旧的 1024 维索引，再改列，最后重建函数
DROP INDEX IF EXISTS idx_images_embedding;

ALTER TABLE images ALTER COLUMN embedding TYPE vector(2048);

-- 重建搜索函数（维度同步更新）
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
