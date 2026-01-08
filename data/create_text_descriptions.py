import json

# File paths
input_file = '/Users/zrz/Desktop/组会/CEMG/data/item_info.json'
output_file = '/Users/zrz/Desktop/组会/CEMG/data/item_text_descriptions.json'

print("Loading item info...")
with open(input_file, 'r', encoding='utf-8') as f:
    item_info = json.load(f)

print(f"Loaded {len(item_info)} items")

# Create text descriptions
print("\nCreating text descriptions...")
item_descriptions = {}
stats = {
    'total': 0,
    'with_categories': 0,
    'with_title': 0,
    'with_both': 0
}

for item_id, info in item_info.items():
    stats['total'] += 1

    title = info.get('title', '').strip()
    categories = info.get('categories', [])

    # Build description parts
    description_parts = []

    # Add categories (join with ' > ')
    if categories and len(categories) > 0:
        # categories is a list of lists, take the first one
        category_path = categories[0] if isinstance(categories[0], list) else categories
        category_text = ' > '.join(category_path)
        description_parts.append(f"Categories: {category_text}")
        stats['with_categories'] += 1

    # Add title
    if title:
        description_parts.append(f"Title: {title}")
        stats['with_title'] += 1

    if len(description_parts) == 2:
        stats['with_both'] += 1

    # Join with newline
    full_description = '\n'.join(description_parts)

    item_descriptions[item_id] = full_description

print(f"Created {len(item_descriptions)} text descriptions")

# Save to JSON file
print("\nSaving to JSON file...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(item_descriptions, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_file}")

# Display statistics
print("\n" + "="*60)
print("TEXT DESCRIPTION STATISTICS")
print("="*60)
print(f"Total items: {stats['total']}")
print(f"Items with categories: {stats['with_categories']} ({stats['with_categories']/stats['total']*100:.2f}%)")
print(f"Items with title: {stats['with_title']} ({stats['with_title']/stats['total']*100:.2f}%)")
print(f"Items with both: {stats['with_both']} ({stats['with_both']/stats['total']*100:.2f}%)")
print("="*60)

# Show a few examples
print("\nExample descriptions:")
print("-"*60)
for i, (item_id, desc) in enumerate(list(item_descriptions.items())[:3]):
    print(f"\nItem ID: {item_id}")
    print(desc)
    if i < 2:
        print("-"*60)
