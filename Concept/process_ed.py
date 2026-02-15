import os
import json
from tqdm import tqdm
import time
import argparse
import re


prompt = """
'''
This task involves identifying the event concept for a given trigger span, including event type, type description and similar examples. Based on the provided sentence and trigger span, generate (1) event type (as fine-grained as possible, matching the span’s context); (2) description (defining the category); (3) examples (illustrating similar triggers)
Ensure the event type precisely reflects the connotation of the trigger span in the sentence.
'''

# Here are some examples:

sentence = "Witnesses say the killing began when government forces shelled the village of Teldau soon after Friday prayers"
trigger_span = "shelled"
event_types = [
    {{
        "event_type": "Attack",
        "description": An Attack Event is defined as a violent physical act causing harm or damage.",
        "examples": "attack, airstrikes, shooting, shot, attacks"
    }},
    {{
        "event_type": "ArtilleryStrike",
        "description": "An ARTILLERY-STRIKE event involves the use of large-caliber guns or missile systems to bombard a specific location, often as part of a military operation.",
        "examples": "shelled, bombarded, pounded, blasted, fired on"
    }},
    {{
        "event_type": "ViolentConflictAction",
        "description": "A VIOLENT-CONFLICT-ACTION event refers to an aggressive act carried out during a conflict, such as warfare or insurgency, involving physical force intended to damage or destroy.",
        "examples": "attacked, shelled, raided, struck, assaulted"
    }}
]

sentence = "Cross-linking CD40 on B cells can lead to homotypic cell adhesion, IL-6 production, and, in combination with cytokines, to Ig isotype switching. Tyrosine kinase activity is increased shortly after engagement of this receptor."
trigger_span = "lead"

event_types = [
    {{
        "event_type": "Positive_regulation",
        "description": "Any process that activates or increases the rate, frequency or extent of a biological process. Biological processes are regulated by many means; examples include the control of gene expression, protein modification or interaction with a protein or substrate molecule.",
        "examples": "lead, lead to, activate, leads, increase"
    }},
    {{
        "event_type": "Induction",
        "description": "A process that initiates or causes a downstream biological event, typically involving molecular or cellular changes initiated by a stimulus or signaling molecule.",
        "examples": "trigger, lead to, cause, initiate, induce"
    }},
    {{
        "event_type": "Signal_transduction",
        "description": "The cellular process in which a signal is conveyed from the exterior to the interior of the cell, resulting in a functional change such as gene expression or cell differentiation.",
        "examples": "activate, signal through, lead to, engage, transmit"
    }}
]

sentence = "The National Carousel Association was founded in the early 1970 ' s by a group of art historians and collectors"
trigger_span = "founded"

event_types = [
    {{
        "event_type": "StartOrganization",
        "description": "A START-ORGANIZATION Event occurs whenever a new ORGANIZATION is created.",
        "examples": "founded, set up, started, form, shape, future, forming, craft, put in place, begin work."
    }},
    {{{{
        "event_type": "OrganizationFormation",
        "description": "An ORGANIZATION-FORMATION event occurs when a new organization is officially established. This includes legal registration or formal recognition.",
        "examples": "incorporated, registered, chartered, founded, established, constituted, formed."
    }},
    {{
        "event_type": "HistoricalResearchAssociationCreation",
        "description": "A HISTORICAL-RESEARCH-ASSOCIATION-CREATION event occurs when a group focused on historical studies, documentation, or academic collaboration is established.",
        "examples": "founded, organized, formed, instituted, assembled, gathered, launched."
    }}
]

sentence = "Nearby, soldiers entered the home of the Bayasi family, the HRW report said"
trigger_span = "entered"
event_types = [
    {{
        "event_type": "Transport",
        "description": "A TRANSPORT Event occurs whenever an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GPE, FACILITY, LOCATION) to another.",
        "examples": "flee, arrive, traveled, moved, fled"
    }},
    {{
        "event_type": "Enter",
        "description": "A ENTER event occurs when an entity, typically a person or group, moves into a specific location or property, often implying access or intrusion.",
        "examples": "entered, stepped into, walked in, accessed, invaded"
    }},
    {{
        "event_type": "PropertyAccess",
        "description": "A PROPERTY-ACCESS event occurs when individuals or groups gain entry to a private or restricted property, sometimes with legal or illegal intent.",
        "examples": "entered, accessed, broke into, trespassed, infiltrated"
    }}
]

# Now please generate event types for the trigger span and sentence. Please provide 3 related event types.
# Only output the value of event_types.
# Please ensure the output is in json format.
sentence = "{sentence}"
trigger_span = "{trigger_span}"
event_types = 
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
        


parser = argparse.ArgumentParser()
base_dir = "dataset_ed"
output_dir = "dataset_ed_query"

task = "ED"
parser.add_argument('--file_dirs', nargs='+', type=str,
                    default=["geneva", "m2e2", "mlee"])
parser.add_argument('--sets', nargs='+', type=str, default=['train','test'])
args = parser.parse_args()
sets = args.sets
file_dirs = args.file_dirs
max_retries = 20
retry_delay = 5

for file_dir in file_dirs:
    for set_type in sets:
        # 修改 输入文件
        # input_file = "dataset/ACE2005/test.jsonl"
        input_file = os.path.join(base_dir, task, file_dir,f"{set_type}_raw.jsonl")
        # 修改 输出文件
        # output_file = "dataset/ACE2005/test_out.jsonl"
        ensure_dir_exists(os.path.join(output_dir, task, file_dir))
        output_file = os.path.join(base_dir, task, file_dir, f"{set_type}_raw_query.jsonl")
        with open(input_file, "r") as f_r, open(output_file, "w") as f_w:
            file_length = get_length(input_file)
            length = 1
            progress = tqdm(total=file_length, ncols=75, desc='processing')
            for raw_line in f_r:
                progress.update(1)
                line = json.loads(raw_line)
                sentence = line["sentence"]
                pred_evt = line["pred_events"]
                gold_evt = []
                span2type = {}
                for evt in line["events"]:  # 将不是golden span的pred span，进行概念化
                    evt_span = evt["trigger"]
                    if evt["type"] != "Event":  # 有event type的是原始golden，没有的是补标的，不属于golden
                        gold_evt.append(evt_span)
                        span2type[evt_span] = evt["type"]
                line["pred_events_concept"] = []
                for evt in pred_evt:
                    if evt in gold_evt:
                        evt_type = span2type[evt]
                    else:
                        evt_type = "Other"
                    prompt = prompt.format(sentence=sentence, trigger_span=evt)
                    line["pred_events_concept"].append({"trigger": evt, "type": evt_type, "query": prompt_extraction})
                f_w.write(json.dumps(line, ensure_ascii=False) + '\n')
                length += 1
