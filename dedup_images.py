"""
dedup_images.py — 像素级图片去重

扫描 Supabase 中所有图片，基于像素 MD5 找到完全相同的图片。
每组重复图中保留最早上传的作为主图，其余标记为 duplicate_of。
两个源文件都保留，只有显示/搜索时去重。

运行: python dedup_images.py
"""

import sys, io, hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

import httpx
from PIL import Image
from supabase_client import get_all_images


def pixel_hash_from_url(url: str, timeout: int = 30) -> str | None:
    """下载图片 → 解码像素 → MD5 哈希。

    返回 None 表示下载或解码失败。
    """
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        # 统一转 RGB 后哈希，忽略格式差异（同图 PNG/JPG 也算重复）
        img = img.convert("RGB")
        return hashlib.md5(img.tobytes()).hexdigest()
    except Exception as e:
        print(f"   ⚠ 下载/解码失败: {url[:80]}... — {e}")
        return None


def find_duplicates():
    """主流程：扫描 → 计算哈希 → 标记重复。"""
    print("=" * 60)
    print("  像素级图片去重检测")
    print("=" * 60)

    # 1. 获取所有图片
    print("\n[1/4] 加载所有图片记录...")
    images = get_all_images()
    if not images:
        print("   没有图片记录")
        return
    print(f"   共 {len(images)} 张图片")

    # 2. 计算每张图的像素哈希
    print("\n[2/4] 计算像素哈希...")
    hash_map = defaultdict(list)  # hash → [image records]
    skipped = 0

    for i, img in enumerate(images):
        url = img.get("file_path", "")
        if not url:
            skipped += 1
            continue

        ph = pixel_hash_from_url(url)
        if ph is None:
            skipped += 1
            continue

        hash_map[ph].append(img)

        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{len(images)}...")

    print(f"   完成: {len(images) - skipped} 张哈希, {skipped} 张跳过")

    # 3. 找出重复组
    print("\n[3/4] 检测重复组...")
    dup_groups = {h: imgs for h, imgs in hash_map.items() if len(imgs) > 1}

    if not dup_groups:
        print("   ✅ 没有发现像素级重复图片")
        return

    print(f"   发现 {len(dup_groups)} 组重复，共 {sum(len(g) for g in dup_groups.values())} 张图片")

    # 4. 标记重复（每组保留最早的一张）
    print("\n[4/4] 标记重复...")
    total_marked = 0

    for ph, group in dup_groups.items():
        # 按上传时间排序，最早的是主图
        group_sorted = sorted(group, key=lambda x: x.get("upload_time", ""))
        primary = group_sorted[0]
        duplicates = group_sorted[1:]

        print(f"\n   哈希: {ph[:12]}...")
        print(f"   主图: ID={primary['id']} \"{primary['file_name']}\" "
              f"({primary.get('uploader','?')} @ {primary.get('upload_time','?')[:10]})")

        for dup in duplicates:
            print(f"   重复: ID={dup['id']} \"{dup['file_name']}\" "
                  f"({dup.get('uploader','?')} @ {dup.get('upload_time','?')[:10]})")

            try:
                from supabase_client import _get_db
                conn = _get_db()
                cur = conn.cursor()
                cur.execute("UPDATE images SET duplicate_of = %s WHERE id = %s", (primary["id"], dup["id"]))
                conn.commit()
                cur.close()
                conn.close()
                total_marked += 1
            except Exception as e:
                print(f"      ❌ 标记失败: {e}")

    print(f"\n{'=' * 60}")
    print(f"  完成: {total_marked} 张重复图片已标记")
    print(f"  它们在搜索中会被隐藏，但文件和记录都保留")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    find_duplicates()
