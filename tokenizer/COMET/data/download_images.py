#!/usr/bin/env python3
"""
Download item images from Amazon URLs and produce an enriched JSON with local paths.

Input:  item_info.json  (key=item_id, value={..., "imUrl": "http://..."})
Output: beauty_image/{item_id}.jpg          (downloaded images)
        item_info_with_image.json           (original fields + "image_path")

Usage:
    python download_images.py \
        --input_json /path/to/item_info.json \
        --output_dir /path/to/beauty_image \
        --output_json /path/to/item_info_with_image.json \
        --workers 32 --timeout 10
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from tqdm import tqdm


def download_one(item_id, url, output_dir, timeout):
    """Download a single image. Returns (item_id, local_path or None, error_msg)."""
    if not url:
        return item_id, None, "empty_url"

    ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        ext = ".jpg"
    local_path = os.path.join(output_dir, f"{item_id}{ext}")

    # Skip if already downloaded
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return item_id, local_path, None

    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 100:  # too small, likely an error page
            return item_id, None, "too_small"
        with open(local_path, "wb") as f:
            f.write(data)
        return item_id, local_path, None
    except (HTTPError, URLError, TimeoutError, OSError) as e:
        return item_id, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to item_info.json")
    parser.add_argument("--output_dir", required=True, help="Directory to save images")
    parser.add_argument("--output_json", required=True, help="Path to output JSON with image paths")
    parser.add_argument("--workers", type=int, default=32, help="Number of download threads")
    parser.add_argument("--timeout", type=int, default=10, help="Download timeout per image (seconds)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r") as f:
        item_info = json.load(f)

    print(f"Total items: {len(item_info)}")

    # Count existing
    urls = {iid: info.get("imUrl", "") for iid, info in item_info.items()}
    has_url = sum(1 for u in urls.values() if u)
    print(f"Items with URL: {has_url}, without URL: {len(urls) - has_url}")

    # Download with thread pool
    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, iid, url, args.output_dir, args.timeout): iid
            for iid, url in urls.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            item_id, local_path, err = future.result()
            if local_path:
                results[item_id] = local_path
            else:
                failed.append((item_id, err))

    print(f"\nSuccess: {len(results)}/{len(item_info)}")
    print(f"Failed:  {len(failed)}/{len(item_info)}")

    # Print failure breakdown
    error_counts = {}
    for _, err in failed:
        key = err if len(err) < 60 else err[:60]
        error_counts[key] = error_counts.get(key, 0) + 1
    if error_counts:
        print("\nFailure reasons:")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {cnt:5d}  {err}")

    # Build output JSON: original info + "image_path"
    output = {}
    for iid, info in item_info.items():
        entry = dict(info)
        if iid in results:
            entry["image_path"] = results[iid]
        else:
            entry["image_path"] = ""
        output[iid] = entry

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_json}")

    # Summary
    coverage = len(results) / len(item_info) * 100
    print(f"Image coverage: {coverage:.1f}%")


if __name__ == "__main__":
    main()
