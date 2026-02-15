import json
from tqdm import tqdm
import re
from multiprocessing import Pool, cpu_count
import time
import mmap
import orjson
from collections import defaultdict

# with open("badwords2.json", "r") as f:
#     bad_words = set(json.load(f))

# Add prepositions
stop_words = {'!', ',', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '.', '/', ':', ';', '<', '=', '>', '?',
              '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', 'that', 'our', 'little', 'herself', 'from', 'hadn',
              'those', 'most', 'why', 'further', 'am', 'i', 'I', 'either', 'or', 'mustn', "must'n", "must'", "must' n", 'who', 'Who', 'her', 'everything', 'don', "don'", "don' t", "don't",
              'be', 'me', 'yours', 'same', 'all', 'own', 'and', 'him', 'these', 'for', 'had', 'too', 'we', 'We', 'so', 'hers',
              'down', 'shouldn', "shouldn'", "shouldn't", "shouldn' t", 'could', 'once', 'isn', 'yet', 'but', 'hasn', "hasn'", "hasn't", "hasn' t", 'anybody', 'because', 'couldn', "couldn'", "couldn't", "couldn' t",
              'something', 'were', 'while', 'us', 'another', 'are', 'you', 'here', 'just', 'themselves', 'against',
              'might', 'only', 'itself', 'must', 'their', 'yourself', 'of', 'about', 'his', 'although', 'them',
              'anything', 'someone', 'has', 'now', 'theirs', 'a', 'in', 'up', 'did', 'weren', "weren'", "weren't", "weren' t", 'between', 'as', 'ours',
              'won', "won'", "won't", "won' t", 'other', 'any', 't', "'t", "' t", 'this', 'none', 'under', 'is', 'such', 'during', 'was', 'at', 'several',
              'whom', 'may', 's', 'nobody', 'again', 'if', 'being', 'the', 'neither', 'much', 'nothing', 'to', 'myself',
              'with', 'been', 'both', 'they', 'They', 'wasn', "wasn'", "wasn' t", "wasn't", 'yourselves', 'unless', 'before', 'also', 'some', 'should',
              'your', 'then', 'below', 'through', 'off', 'on', 'over', 'its', 'my', 'an', 'mine', 'even', 'when',
              'after', 'doesn', "doesn'", "doesn't", "doesn' t", 'no one', 'haven', 'since', 'many', 'everyone', 'very', 'more', 'whose', 'didn', "didn'", "didn't", "didn' t",
              'having', 'do', 'there', 'he', 'He', 'anyone', 'above', 'each', 'whether', 'can', 'would', 'aren', 'few', 'it',
              'how', 'into', 'every', 'what', 'she', 'not', 'ourselves', 'shall', 'needn', "needn'", "needn't", "needn' t", 'somebody', 'mightn', "mightn'", "mightn' t", "mightn't",
              'where', 'himself', 'than', 'by', 'doing', 'no', 'which', 'wouldn', "wouldn'", "wouldn't", "wouldn' t", 'until', 'will', 'shan', 'have',
              'does', 'everybody', "each other", "one another", "”", "“"}




def find_numbers(text):
    pattern = r"""
        -?\s*\d+       # Match negative numbers (e.g., "- 3" or "-3")
        (?:            # Match decimals or fractions
            \.\d+      # Decimal (e.g., 1.23)
            |          # Or
            \s*\/\s*\d+  # Fraction (e.g., 1 / 2)
        )?             # Decimal or fraction is optional
        \s*%?          # Optional percent sign (e.g., "50 %")
    """
    return re.findall(pattern, text, re.VERBOSE)


def has_overlap(entity, existing_entities):
    """Check whether the entity overlaps with existing entities (case-insensitive)"""
    entity_lower = entity.lower()
    for existing_ent in existing_entities:
        existing_lower = existing_ent.lower()
        # Check full containment or partial overlap
        if (entity_lower in existing_lower) or (existing_lower in entity_lower):
            return True
        # Check word-level overlap separated by spaces
        entity_parts = set(entity_lower.split())
        existing_parts = set(existing_lower.split())
        if entity_parts & existing_parts:
            return True
    return False


def find_entity_in_sentence(entity, sentence, pattern_cache):
    """Find an entity using a precompiled regular expression"""
    if entity not in pattern_cache:
        pattern_cache[entity] = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)

    match = pattern_cache[entity].search(sentence)
    if match:
        matched_text = match.group()
        if len(matched_text) == len(entity) or matched_text.lower() == entity.lower():
            return matched_text
    return None


def process_line(args):
    """Function to process a single line of data for multiprocessing"""
    line, entity_patterns, first_char_index = args
    try:
        data = orjson.loads(line)
        sentence = data["sentence"]
        existing_entities = {ent["name"] for ent in data["entities"] if ent["name"] not in stop_words}
        new_entities = []

        # Quickly filter candidate entities
        first_chars = {w[0].lower() for w in sentence.split() if w}
        candidates = [ent for c in first_chars for ent in first_char_index.get(c, [])]

        # Check each candidate entity
        for entity in candidates:
            if has_overlap(entity, existing_entities):
                continue
            matched_entity = find_entity_in_sentence(entity, sentence, entity_patterns)
            if matched_entity and not has_overlap(matched_entity, existing_entities):
                new_entities.append({"pos": [], "name": matched_entity, "type": "Entity"})
                existing_entities.add(matched_entity)

        # Append number entity annotations
        numbers = find_numbers(sentence)
        for num in numbers:
            num = num.strip()
            if num and not has_overlap(num, existing_entities):
                new_entities.append({"pos": [], "name": num, "type": "NumberEntityNew"})
                existing_entities.add(num)

        # Merge and filter stopwords
        filtered_entities = [
            ent for ent in data["entities"] + new_entities
        ]
        data["entities"] = filtered_entities

        # Generate code
        unique_entities = {ent["name"] for ent in filtered_entities}
        code = "results = [\n" + "".join(f'\tEntity("{ent}"),\n' for ent in unique_entities)
        code = code[:-2] + "\n]" if unique_entities else "results = []"
        data["code"] = code

        return orjson.dumps(data).decode('utf-8')
    except Exception as e:
        print(f"Error processing line: {e}")
        return None


def process_file(input_file, output_file):
    # Load entity data
    with open("entity_filter_3more.json", "rb") as f_r:
        entity_dict = orjson.loads(f_r.read())
        sorted_entities = [word for word, freq in sorted(entity_dict.items(), key=lambda x: x[1], reverse=True)]

    # Build entity index
    first_char_index = defaultdict(list)
    for entity in sorted_entities:
        if entity:
            first_char_index[entity[0].lower()].append(entity)

    # Precompile regular expressions for common entities
    common_entities = sorted_entities[:10000]  # Precompile the top 10,000 most common entities
    entity_patterns = {e: re.compile(r'\b' + re.escape(e) + r'\b', re.IGNORECASE)
                       for e in common_entities}

    # Prepare multiprocessing
    num_workers = max(1, cpu_count() - 1)
    pool = Pool(num_workers)

    # Get total line count for the progress bar
    with open(input_file, "rb") as f:
        total_lines = sum(1 for _ in f)

    # Process the file
    start_time = time.time()
    processed_count = 0

    with open(input_file, "rb") as f_r, open(output_file, "wb") as f_w:
        # Use memory mapping to speed up reading
        mm = mmap.mmap(f_r.fileno(), 0, access=mmap.ACCESS_READ)

        # Use tqdm to display progress
        with tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
            batch_size = 1000
            batch = []

            for line in iter(mm.readline, b""):
                batch.append((line, entity_patterns, first_char_index))

                if len(batch) >= batch_size:
                    results = pool.imap(process_line, batch, chunksize=100)
                    for result in results:
                        if result:
                            f_w.write(result.encode('utf-8') + b"\n")
                            processed_count += 1

                    # Update progress and estimate remaining time
                    elapsed = time.time() - start_time
                    lines_per_sec = processed_count / elapsed if elapsed > 0 else 0
                    remaining = (total_lines - processed_count) / lines_per_sec if lines_per_sec > 0 else 0

                    pbar.set_postfix({
                        "speed": f"{lines_per_sec:.1f} lines/s",
                        "remaining": f"{remaining / 3600:.1f}h"
                    })
                    pbar.update(len(batch))
                    batch = []

            # Process remaining lines that are fewer than one batch
            if batch:
                results = pool.imap(process_line, batch, chunksize=100)
                for result in results:
                    if result:
                        f_w.write(result.encode('utf-8') + b"\n")
                pbar.update(len(batch))

    pool.close()
    pool.join()

    print(f"Processing completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    input_file = "merge/train.json"  # Merge all NER files
    output_file = "merge/train-addnotation.json"
    process_file(input_file, output_file)
