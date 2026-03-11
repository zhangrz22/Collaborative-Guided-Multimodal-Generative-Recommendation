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
    'with_brand': 0,
    'with_price': 0,
    'with_description': 0,
    'with_all': 0
}

for item_id, info in item_info.items():
    stats['total'] += 1

    title = info.get('title', '').strip()
    categories = info.get('categories', [])
    brand = info.get('brand', '').strip()
    price = info.get('price', '')
    description = info.get('description', '').strip()

    # Build description parts
    description_parts = []

    # Add title (第一个)
    if title:
        description_parts.append(f"Title: {title}")
        stats['with_title'] += 1

    # Add brand (第二个)
    if brand:
        description_parts.append(f"Brand: {brand}")
        stats['with_brand'] += 1

    # Add price (第三个)
    if price:
        description_parts.append(f"Price: ${price}")
        stats['with_price'] += 1

    # Add description (第四个)
    if description:
        description_parts.append(f"Description: {description}")
        stats['with_description'] += 1

    # Add categories (最后一个)
    if categories and len(categories) > 0:
        # categories is a list of lists, take the first one
        category_path = categories[0] if isinstance(categories[0], list) else categories
        category_text = ' > '.join(category_path)
        description_parts.append(f"Categories: {category_text}")
        stats['with_categories'] += 1

    if categories and title and brand and price and description:
        stats['with_all'] += 1

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
print(f"Items with brand: {stats['with_brand']} ({stats['with_brand']/stats['total']*100:.2f}%)")
print(f"Items with price: {stats['with_price']} ({stats['with_price']/stats['total']*100:.2f}%)")
print(f"Items with description: {stats['with_description']} ({stats['with_description']/stats['total']*100:.2f}%)")
print(f"Items with all fields: {stats['with_all']} ({stats['with_all']/stats['total']*100:.2f}%)")
print("="*60)

# Show a few examples
print("\nExample descriptions:")
print("-"*60)
for i, (item_id, desc) in enumerate(list(item_descriptions.items())[:3]):
    print(f"\nItem ID: {item_id}")
    print(desc)
    if i < 2:
        print("-"*60)
