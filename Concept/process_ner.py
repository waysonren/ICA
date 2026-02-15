import os
import json
from tqdm import tqdm
import time
import argparse
import re

prompt = """
'''
This task involves generating the entity concept for a given entity span, including its entity type, type description, and similar examples. Based on the provided sentence and entity span, generate the corresponding: (1)entity_type (as fine-grained as possible, matching the span’s context); (2)description (defining the category); (3)examples (illustrating similar entities).
Ensure the entity type precisely reflects the connotation of the entity span in the sentence.
'''

# Here are some examples:
# Here is a example:
sentence = "Vinken will join the board as a nonexecutive director Nov 29."
entity_span = "Vinken"

entity_types = [
    {{
        "entity_type": "Entrepreneur",
        "description": "person who owns and operates a business.",
        "examples": "Steve Jobs, Zuckerberg, Bill Gates, Thomas Edison, Larry Page."
    }},
    {{
        "entity_type": "Businessman",
        "description": "person engaged in commercial or industrial business activities.",
        "examples": "Warren Buffett, Richard Branson, Elon Musk, Jeff Bezos, Sam Walton."
    }},
    {{
        "entity_type": "Person",
        "description": "Each distinct person or set of people mentioned in a document refers to an entity of type Person.",
        "examples": "Donald Trump, children, women, user, patient, Trump, President Trump, Barack Obama, people."
    }}
]


sentence = "Nevertheless , the main outcome of the reported experience is the proposed redesign approach for part consolidation using metal AM ."
entity_span = "metal AM"

entity_types = [
    {{
        "entity_type": "ManufacturingProcess",
        "description": "Refers to a specific technique, method, or procedure used in the production of goods.",
        "examples": "AM, additive manufacturing, SLM, 3D printing, fabrication, FFF, manufacturing, 3D printed, LPBF, additively manufactured"
    }},
    {{
        "entity_type": "AdvancedMetalFabrication",
        "description": "Encompasses modern and innovative techniques for manufacturing metal components.",
        "examples": "metal AM, powder metallurgy, metal injection molding, laser metal deposition, cold spray additive manufacturing"
    }},
    {{
        "entity_type": "IndustrialManufacturingTechnology",
        "description": "Refers to technologies used in industrial-scale manufacturing processes, particularly for metals.",
        "examples": "CNC machining of metals, metal casting, metal stamping, metal forging, metal extrusion"
    }}
]

sentence = "Comparison with alkaline phosphatases and 5 - nucleotidase"
entity_span = "alkaline phosphatases"

entity_types = [
    {{
        "entity_type": "Gene",
        "description": "Names of genes and gene products (like proteins). Genetic sequences when referred to by specific names or identifiers. Occasionally, mentions of gene families or complexes, if they are referred to by a specific name.",
        "examples": "insulin, Ras, Sp1, p53, AP - 1, CAT, NF - kappaB, JNK, MAPK, PKC"
    }},
    {{
        "entity_type": "Enzyme",
        "description": "Refers to proteins that catalyze biochemical reactions, often named for their substrate or reaction type.",
        "examples": "alkaline phosphatase, DNA polymerase, lactase, catalase, amylase, lipase, 5'-nucleotidase"
    }},
    {{
        "entity_type": "Biomarker",
        "description": "Refers to measurable substances whose presence or concentration indicates biological processes, conditions, or diseases.",
        "examples": "alkaline phosphatase, PSA, troponin, creatinine kinase, C-reactive protein"
    }},
]

sentence = "Switched on the TV for {{@Sky News@}} at 6pm"
entity_span = "Sky News"

entity_types = [
    {{
        "entity_type": "Product",
        "description": "This might involve names of specific products, brands, or goods.",
        "examples": "YouTube, Poshmark, Change org Россия, Google News, DistroKid, Yahoo, Netflix, Daily Mail Online, Etsy, ESPN+"
    }},
    {{
        "entity_type": "NewsChannel",
        "description": "Refers to television or digital platforms that broadcast news content.",
        "examples": "Sky News, BBC News, CNN, Fox News, Al Jazeera, MSNBC, CNBC, Bloomberg TV, France 24, CGTN"
    }},
    {{
        "entity_type": "MediaBrand",
        "description": "Recognizable names of media organizations that produce and distribute content.",
        "examples": "Sky News, The New York Times, Reuters, The Guardian, The Washington Post, BBC, CNN, NBC, CBS, ABC"
    }},
]


# Now please generate entity types for the entity span and sentence. Please provide 3 related entity types.
# Only output the value of entity_types.
# Please ensure the output is in json format.
sentence = "{sentence}"
entity_span = "{entity_span}"
entity_types = 
"""


def dump_json_file(obj, file):
    with open(file, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)


def get_length(file):
    f_all_data = open(file, 'r').readlines()
    return len(f_all_data)


def ensure_dir_exists(directory_path):
    # Check if the file exists
    os.makedirs(directory_path, exist_ok=True)
        



# 修改 system_prompt
# system_prompt = "You are a highly skilled assistant at identifying entity type for entity span based on provided sentences. Below is an example and a query for conceptualization. Please complete it in the form of Python class code."

parser = argparse.ArgumentParser()
base_dir = "dataset_ner"
output_dir = "dataset_ner_query"

task = "NER"
parser.add_argument('--file_dirs', nargs='+', type=str,
                    default=["bc2gm", "Polyglot-NER", "WikiNeural", "FabNER"])
parser.add_argument('--sets', nargs='+', type=str, default=['train'])
args = parser.parse_args()
sets = args.sets
file_dirs = args.file_dirs
max_retries = 20
retry_delay = 5

for file_dir in file_dirs:
    for set_type in sets:
        # 修改 输入文件
        # input_file = "dataset/ACE2005/test.jsonl"
        input_file = os.path.join(base_dir, task, file_dir, f"{set_type}_raw.jsonl")
        # 修改 输出文件
        # output_file = "dataset/ACE2005/test_out.jsonl"
        ensure_dir_exists(os.path.join(output_dir, task, file_dir))
        output_file = os.path.join(output_dir, task, file_dir, f"{set_type}_raw_query.jsonl")
        with open(input_file, "r") as f_r, open(output_file, "w") as f_w:
            file_length = get_length(input_file)
            length = 1
            progress = tqdm(total=file_length, ncols=75, desc='processing')
            for raw_line in f_r:
                progress.update(1)
                line = json.loads(raw_line)
                sentence = line["sentence"]
                pred_ent = line["pred_entities"]
                gold_ent = []
                span2type = {}
                for ent in line["entities"]:  # 将不是golden span的pred span，进行概念化
                    ent_span = ent["name"]
                    if ent["pos"]:  # 有pos参数的是原始golden，没有pos参数的是补标的，不属于golden
                        gold_ent.append(ent_span)
                        span2type[ent_span] = ent["type"]
                for ent in pred_ent:
                    prompt_extraction = prompt.format(sentence=sentence, entity_span=ent)
                    if ent in gold_ent:
                        ent_type = span2type[ent]
                    else:
                        ent_type = "Other"
                    line["pred_entities_concept"].append({"name": ent, "type": ent_type, "query": prompt_extraction})
                f_w.write(json.dumps(line, ensure_ascii=False) + '\n')
                length += 1
