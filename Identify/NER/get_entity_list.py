import json
from collections import defaultdict

target_dataset = {}  # Fill in all NER datasets, such as "Broad Twitter", "CoNLL 2003", etc.

# Use defaultdict to count the occurrences of each entity
trigger_counts = defaultdict(int)

for dataset in target_dataset:
    for set_type in ['train']:
        print(f"Processing {dataset}/{set_type}.json")
        try:
            with open(f"{dataset}/{set_type}.json", "r") as f_r:
                data = json.load(f_r)
                for line in data:
                    for evt in line["entities"]:
                        trigger = evt["name"]
                        trigger_counts[trigger] += 1
        except FileNotFoundError:
            print(f"Warning: File not found - {dataset}/{set_type}.json")
            continue
        except json.JSONDecodeError:
            print(f"Warning: JSON decode error in {dataset}/{set_type}.json")
            continue

# Sort by frequency in descending order, then by length in descending order
sorted_triggers = sorted(
    trigger_counts.items(),
    key=lambda x: (-x[1], -len(x[0]), x[0])
)

# Convert to dictionary format
result = {trigger: count for trigger, count in sorted_triggers}

# Write to JSON file
with open("entity_frequencies.json", "w", encoding='utf-8') as f_w:
    json.dump(result, f_w, ensure_ascii=False, indent=4)

print(f"Successfully saved trigger frequencies to entity_frequencies.json")
print(f"Total unique triggers: {len(result)}")
