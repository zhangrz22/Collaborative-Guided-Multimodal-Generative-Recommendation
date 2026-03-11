input_file = '/Users/zrz/Desktop/组会/CEMG/data/interaction_sequences.txt'
output_file = '/Users/zrz/Desktop/组会/CEMG/data/interaction_sequences_truncated.txt'

total = 0
truncated = 0

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    header = f_in.readline()
    f_out.write(header)

    for line in f_in:
        user_id, sequence = line.strip().split('\t')
        items = sequence.split()
        total += 1

        if len(items) > 50:
            items = items[-50:]
            truncated += 1

        f_out.write(f"{user_id}\t{' '.join(items)}\n")

print(f"Total users: {total}")
print(f"Truncated: {truncated} ({truncated/total*100:.2f}%)")
print(f"Output saved to {output_file}")
