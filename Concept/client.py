import json
import os
import argparse
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import deque
import re

# Configuration parameters
BATCH_SIZE = 5  # Number of lines processed per batch
MAX_WORKERS = 5  # Number of concurrent threads


def validate_response(response):
    """Validate whether the response conforms to the expected JSON format"""
    try:
        data = json.loads(response)
        if not isinstance(data, list):
            return False
        if len(data) != 3:
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if not (all(key in item for key in ["entity_type", "description", "examples"]) or
                    all(key in item for key in ["event_type", "description", "examples"])):
                return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def query_qwen(prompt, max_tokens=512, url="http://127.0.0.1:8006/generate"):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "max_tokens": max_tokens}
    response = requests.post(url, headers=headers, json=data)
    return response.json()


def batch_process_lines(lines, query_key, url, max_retries=20, retry_delay=5):
    # Prepare all queries
    all_queries = []
    line_refs = []  # Record the corresponding line and entity index for each query
    has_concept_flags = []  # Record whether each query already contains a concept field

    for line_idx, line in enumerate(lines):
        line_data = json.loads(line)
        for ent_idx, ent in enumerate(line_data[query_key]):
            if "concept" in ent:  # Skip query if concept field already exists
                all_queries.append(None)  # Placeholder
                has_concept_flags.append(True)
            else:
                all_queries.append(ent["query"])
                has_concept_flags.append(False)
            line_refs.append((line_idx, ent_idx))

    # Batch process queries
    results = [None] * len(all_queries)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for i, (query, has_concept) in enumerate(zip(all_queries, has_concept_flags)):
            if has_concept:  # Skip queries that already have concept
                continue
            future = executor.submit(
                query_qwen_with_retry,
                query,
                512,
                url,
                max_retries,
                retry_delay
            )
            futures.append((i, future))

        for i, future in tqdm(futures, desc="Processing queries", leave=False):
            try:
                results[i] = future.result()
            except Exception as e:
                print(f"Error processing query: {e}")
                results[i] = "Error: Failed to get response"

    # Assign results back to original lines
    processed_lines = [json.loads(line) for line in lines]
    for (line_idx, ent_idx), result, has_concept in zip(line_refs, results, has_concept_flags):
        if has_concept:  # Skip entities that already have concept
            continue
        processed_lines[line_idx][query_key][ent_idx]["result"] = result
        try:
            processed_lines[line_idx][query_key][ent_idx]["concept"] = json.loads(result)
        except json.JSONDecodeError:
            # If the result is not valid JSON, keep the original error message
            processed_lines[line_idx][query_key][ent_idx]["concept"] = result

    return processed_lines


def query_qwen_with_retry(prompt, max_tokens, url, max_retries, retry_delay):
    for attempt in range(max_retries):
        try:
            result = query_qwen(prompt, max_tokens, url)
            response = result.get("response", "")
            pattern = r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]'
            matches = re.search(pattern, response)
            if matches:
                response = matches.group(0)

            # Validate response format
            if "Error: " not in response and validate_response(response):
                return response

            print(f"Invalid response format or server error, Retrying... ({attempt + 1}/{max_retries})")
            print(f"Received response: {response}")  # Print received response for debugging
        except Exception as e:
            print(f"Error: {e}. Retrying... ({attempt + 1}/{max_retries})")
        time.sleep(retry_delay)

    # Return default structure if all retries fail
    default_response = json.dumps([
        {"entity_type": "Error", "description": "Failed to get valid response", "examples": []},
        {"entity_type": "Error", "description": "Failed to get valid response", "examples": []},
        {"entity_type": "Error", "description": "Failed to get valid response", "examples": []}
    ])
    return {"response": default_response}


def process_file(input_file, output_file, query_key, start_num, end_num, url):
    with open(input_file, "r") as f_r, open(output_file, "a", buffering=1) as f_w:
        # Skip initial lines
        for _ in range(start_num):
            next(f_r)

        line_buffer = deque()
        total_processed = start_num

        # Use progress bar
        with tqdm(total=end_num - start_num,
                  desc=f"Processing {os.path.basename(input_file)}") as pbar:
            for line in f_r:
                line_buffer.append(line)

                # Process when buffer reaches batch size
                if len(line_buffer) >= BATCH_SIZE or total_processed + len(line_buffer) >= end_num:
                    processed_lines = batch_process_lines(line_buffer, query_key, url)
                    for pline in processed_lines:
                        f_w.write(json.dumps(pline, ensure_ascii=False) + '\n')

                    pbar.update(len(line_buffer))
                    total_processed += len(line_buffer)
                    line_buffer.clear()

                if total_processed >= end_num:
                    break

            # Process remaining lines
            if line_buffer:
                processed_lines = batch_process_lines(line_buffer, query_key, url)
                for pline in processed_lines:
                    f_w.write(json.dumps(pline, ensure_ascii=False) + '\n')
                pbar.update(len(line_buffer))


def get_length(file):
    f_all_data = open(file, 'r').readlines()
    return len(f_all_data)


# Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="NER")
    parser.add_argument('--base_dir', type=str, default="dataset_knowcoder_v21_concept")
    parser.add_argument('--output_dir', type=str, default="dataset_knowcoder_v21_concept_qwen")
    parser.add_argument('--file_dirs', nargs='+', type=str,
                        default=['WikiANN', 'MIT Restaurant', 'MIT Movie', 'MultiNERD', 'Ontonotes 5'])
    parser.add_argument('--sets', nargs='+', type=str, default=['train'])
    parser.add_argument('--start_num', type=int, default=0)
    parser.add_argument('--end_num', type=int, default=2000000)
    parser.add_argument('--port', type=str, default="8006")
    parser.add_argument('--input_suffix', type=str, default="query")
    parser.add_argument('--output_suffix', type=str, default="align")
    parser.add_argument('--query_key', type=str, default="entities")
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--max_workers', type=int, default=5)

    args = parser.parse_args()
    task = args.task
    BATCH_SIZE = args.batch_size
    MAX_WORKERS = args.max_workers
    url = f"http://127.0.0.1:{args.port}/generate"

    for file_dir in args.file_dirs:
        for set_type in args.sets:
            input_file = os.path.join(args.base_dir, task, file_dir,
                                      f"{set_type}_{args.input_suffix}.jsonl")
            output_dir_path = os.path.join(args.output_dir, task, file_dir)
            end_num = args.end_num
            file_length = get_length(input_file)
            if end_num > file_length:
                end_num = file_length
            os.makedirs(output_dir_path, exist_ok=True)
            output_file = os.path.join(output_dir_path,
                                       f"{set_type}_{args.output_suffix}.jsonl")

            print(f"Processing {input_file}...")
            process_file(input_file, output_file,
                         args.query_key, args.start_num, end_num, url)
            print(f"Finished processing. Results saved to {output_file}")
