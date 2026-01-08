import json
from collections import defaultdict

# File paths
reviews_file = '/Users/zrz/Desktop/组会/CEMG/data/reviews_Beauty_5.json'
meta_file = '/Users/zrz/Desktop/组会/CEMG/data/meta_Beauty.json'
output_dir = '/Users/zrz/Desktop/组会/CEMG/data/'

# Step 1: Read reviews and collect unique users and items
print("Reading reviews file...")
user_items = defaultdict(list)  # {reviewerID: [(asin, unixReviewTime), ...]}
unique_users = set()
unique_items = set()

with open(reviews_file, 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line.strip())
        reviewer_id = review['reviewerID']
        asin = review['asin']
        unix_time = review['unixReviewTime']

        unique_users.add(reviewer_id)
        unique_items.add(asin)
        user_items[reviewer_id].append((asin, unix_time))

print(f"Found {len(unique_users)} unique users")
print(f"Found {len(unique_items)} unique items")

# Step 2: Create mappings
print("\nCreating mappings...")
user_mapping = {user: idx for idx, user in enumerate(sorted(unique_users))}
item_mapping = {item: idx for idx, item in enumerate(sorted(unique_items))}

# Save user mapping
print("Saving user mapping...")
with open(output_dir + 'user_mapping.txt', 'w', encoding='utf-8') as f:
    f.write("original_reviewerID\tmapped_user_id\n")
    for user, idx in sorted(user_mapping.items(), key=lambda x: x[1]):
        f.write(f"{user}\t{idx}\n")

# Save item mapping
print("Saving item mapping...")
with open(output_dir + 'item_mapping.txt', 'w', encoding='utf-8') as f:
    f.write("original_asin\tmapped_item_id\n")
    for item, idx in sorted(item_mapping.items(), key=lambda x: x[1]):
        f.write(f"{item}\t{idx}\n")

# Step 3: Create interaction sequences
print("\nCreating interaction sequences...")
with open(output_dir + 'interaction_sequences.txt', 'w', encoding='utf-8') as f:
    f.write("user_id\titem_sequence\n")

    for reviewer_id in sorted(user_mapping.keys()):
        # Get all interactions for this user
        interactions = user_items[reviewer_id]

        # Sort by time (oldest to newest)
        interactions.sort(key=lambda x: x[1])

        # Map to new IDs
        mapped_user_id = user_mapping[reviewer_id]
        mapped_item_sequence = [str(item_mapping[asin]) for asin, _ in interactions]

        # Write to file
        f.write(f"{mapped_user_id}\t{' '.join(mapped_item_sequence)}\n")

print("\nProcessing complete!")
print(f"Output files created in {output_dir}:")
print("  - user_mapping.txt")
print("  - item_mapping.txt")
print("  - interaction_sequences.txt")
