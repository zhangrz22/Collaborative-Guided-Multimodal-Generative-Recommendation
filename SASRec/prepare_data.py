import json

input_file = '/Users/zrz/Desktop/组会/CEMG/data/interaction_sequences_truncated.txt'
output_file = '/Users/zrz/Desktop/组会/CEMG/SASRec/data/Beauty/Beauty.inter.json'

interactions = {}

with open(input_file, 'r') as f:
    next(f)  # Skip header
    for line in f:
        user_id, sequence = line.strip().split('\t')
        items = [int(x) for x in sequence.split()]
        interactions[user_id] = items

with open(output_file, 'w') as f:
    json.dump(interactions, f)

print(f"Converted {len(interactions)} users to {output_file}")
