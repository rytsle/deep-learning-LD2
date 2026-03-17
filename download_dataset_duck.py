"""
Download images using DDGS (formerly duckduckgo_search).
Uses random delays to avoid rate limiting.

Requirements:
    pip install ddgs pillow requests

Usage:
    python download_dataset_duck.py
"""

import time
import random
import requests
from pathlib import Path
from ddgs import DDGS

# --- Config ---
OUTPUT_DIR = Path(__file__).parent / "custom_data"

CLASS_QUERIES = {
    "battery": {
        "queries": [
            "AA battery white background product photo",
            "AAA battery isolated studio photo",
            "lithium battery single white background",
            "rechargeable battery product photo",
            "9V battery white background",
        ],
        "target": 100,
        "start_index": 99,
    },
    "light_switch": {
        "queries": [
            "wall light switch white background",
            "electric light switch isolated",
            "on off switch plate close up",
        ],
        "target": 100,
        "start_index": 99,
    },
}


def download_class(class_name, queries, target, start_index):
    out_dir = OUTPUT_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    seen_urls = set()

    for query in queries:
        if saved >= target:
            break

        print(f"  Query: '{query}' (saved={saved})...")
        time.sleep(random.uniform(2, 5))

        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=target - saved + 10))
        except Exception as e:
            print(f"  Warning: query failed — {e}, waiting 10s...")
            time.sleep(10)
            continue

        for r in results:
            if saved >= target:
                break

            url = r.get("image")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            file_number = start_index + saved
            dest = out_dir / f"{class_name}_{file_number:04d}.jpg"
            try:
                resp = requests.get(url, timeout=6)
                if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                    dest.write_bytes(resp.content)
                    saved += 1
            except Exception:
                pass

            time.sleep(random.uniform(0.2, 0.6))

    return saved


def main():
    print(f"Output: {OUTPUT_DIR.resolve()}\n")

    summary = {}
    for class_name, config in CLASS_QUERIES.items():
        print(f"=== {class_name} (target={config['target']}, starting from index {config['start_index']}) ===")
        count = download_class(
            class_name,
            config["queries"],
            config["target"],
            config["start_index"],
        )
        summary[class_name] = count
        print(f"  Done: {count} images saved\n")
        time.sleep(random.uniform(5, 10))

    print("=== Summary ===")
    for class_name, count in summary.items():
        cfg = CLASS_QUERIES[class_name]
        start = cfg["start_index"]
        end = start + count - 1
        print(f"  {class_name}/: {count} images ({class_name}_{start:04d}.jpg → {class_name}_{end:04d}.jpg)")


if __name__ == "__main__":
    main()