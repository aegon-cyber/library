-- 像素级去重：加 duplicate_of 列
ALTER TABLE images ADD COLUMN IF NOT EXISTS duplicate_of INTEGER REFERENCES images(id);

-- 更新 match_images：只返回非重复的（主图）
DROP FUNCTION IF EXISTS match_images(vector,double precision,integer,text,text,text);

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
    source_file      text,
    source_url       text,
    duplicate_of     int,
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
        i.source_file,
        i.source_url,
        i.duplicate_of,
        1 - (i.embedding <=> query_embedding) as similarity
    from images i
    where i.duplicate_of is null  -- 排除重复图
        and case
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
