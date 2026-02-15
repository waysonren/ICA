import json
from collections import defaultdict


def process_line(line, f_w):
    line["from_file"] = dataset
    sentence = line["sentence"]
    existing_triggers = {evt["trigger"] for evt in line["events"]}
    new_triggers = []
    for trigger in triggers:
        if trigger in existing_triggers:
            continue
        if f' {trigger} ' in f' {sentence} ':  # Ensure exact matching
            line["events"].append({"pos": [], "trigger": trigger, "type": "Event"})
            existing_triggers.add(trigger)  # Avoid duplicate additions
            new_triggers.append(trigger)
    line["new_triggers"] = new_triggers
    f_w.write(json.dumps(line, ensure_ascii=False) + '\n')


target_dataset = {} # Fill in all ED datasets, such as "ACE 2005", "CASIE", etc.


with open("trigger_filter.json", "r") as f:
    trigger_filter = json.load(f)
triggers = trigger_filter.keys()
for set_type in ['train']:
    with open(f"merge/{set_type}.json", "w") as f_w:
        for dataset in target_dataset:
            print(f"Processing {dataset}/{set_type}.json")
            try:
                with open(f"{dataset}/{set_type}.json", "r") as f_r:
                    for raw_line in f_r:
                        line = json.loads(raw_line)
                        process_line(line, f_w)
            except FileNotFoundError:
                print(f"Warning: File not found - {dataset}/{set_type}.json")
                continue
            except json.JSONDecodeError:
                print(f"Warning: JSON decode error in {dataset}/{set_type}.json")
                continue