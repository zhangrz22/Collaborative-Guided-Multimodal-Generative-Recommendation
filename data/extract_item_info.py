import json
import ast
from collections import defaultdict

# File paths
meta_file = '/Users/zrz/Desktop/组会/CEMG/data/meta_Beauty.json'
item_mapping_file = '/Users/zrz/Desktop/组会/CEMG/data/item_mapping.txt'
output_file = '/Users/zrz/Desktop/组会/CEMG/data/item_info.json'

# Step 1: Load item mapping (asin -> mapped_id)
print("Loading item mapping...")
asin_to_id = {}
with open(item_mapping_file, 'r', encoding='utf-8') as f:
    next(f)  # Skip header
    for line in f:
        asin, mapped_id = line.strip().split('\t')
        asin_to_id[asin] = int(mapped_id)

print(f"Loaded {len(asin_to_id)} item mappings")

# Step 2: Read meta data and extract required fields
print("\nReading meta data...")
item_info = {}
stats = {
    'total': 0,
    'title_missing': 0,
    'categories_missing': 0,
    'imUrl_missing': 0,
    'all_present': 0
}

with open(meta_file, 'r', encoding='utf-8') as f:
    for line in f:
        meta = ast.literal_eval(line.strip())
        asin = meta.get('asin')

        # Only process items that are in our mapping
        if asin in asin_to_id:
            mapped_id = asin_to_id[asin]
            stats['total'] += 1

            # Extract only title, categories, imUrl
            title = meta.get('title', '')
            categories = meta.get('categories', [])
            imUrl = meta.get('imUrl', '')

            # Track missing fields
            if not title:
                stats['title_missing'] += 1
            if not categories:
                stats['categories_missing'] += 1
            if not imUrl:
                stats['imUrl_missing'] += 1
            if title and categories and imUrl:
                stats['all_present'] += 1

            # Store with mapped ID as key
            item_info[str(mapped_id)] = {
                'title': title,
                'categories': categories,
                'imUrl': imUrl
            }

print(f"Processed {stats['total']} items from meta data")

# Step 3: Save to JSON file
print("\nSaving item info to JSON...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(item_info, f, ensure_ascii=False, indent=2)

print(f"Saved {len(item_info)} items to {output_file}")

# Step 4: Calculate and display statistics
print("\n" + "="*60)
print("MISSING DATA STATISTICS")
print("="*60)
print(f"Total items: {stats['total']}")
print(f"\nTitle missing: {stats['title_missing']} ({stats['title_missing']/stats['total']*100:.2f}%)")
print(f"Categories missing: {stats['categories_missing']} ({stats['categories_missing']/stats['total']*100:.2f}%)")
print(f"ImUrl missing: {stats['imUrl_missing']} ({stats['imUrl_missing']/stats['total']*100:.2f}%)")
print(f"\nAll fields present: {stats['all_present']} ({stats['all_present']/stats['total']*100:.2f}%)")
print("="*60)
