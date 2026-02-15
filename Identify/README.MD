
# Identify

This module builds candidate lists and cross-annotated datasets for NER and ED, then prepares data for supervised fine-tuning (SFT).

## Environment

- Python 3.9+
- Install required packages:
  - `orjson`, `tqdm`
  - `nltk` 

## Data Layout

Place source datasets under this directory. Expected structure follows what the scripts read:

- NER datasets: folders like `AnatEM`, `bc2gm`, `GENIA`, `WikiNeural`, etc.
- ED datasets: folders like `ACE 2005`, `CASIE`, `GLEN`, `PHEE-new`, `WikiEvents`


## NER Workflow

1) Build entity frequency list

```bash
python NER/get_entity_list.py
```

Output: `entity_frequencies.json`

2) Filter entities

```bash
python NER/filter_entity.py
```

Output: `entity_filter_3more.json`  

3) Cross-annotation

```bash
python NER/add_annotation.py
```

Inputs:
- `merge/train.json` (merged upstream NER data)
- `entity_filter_3more.json` (filtered entity list)

Output: `merge/train-addnotation.json` (entities augmented by cross-annotation)

## ED Workflow

1) Build trigger frequency list

```bash
python ED/get_trigger_list.py
```

Output: `trigger_frequencies.json`

2) Filter triggers

`ED/filter_triggers.py` provides a `TriggerWordFilter` class. Use it programmatically:


Output: `trigger_filter.json`

3) Cross-annotation

```bash
python ED/add_annotation.py
```

Inputs:
- `merge/train.json` (merged upstream ED data)
- `trigger_filter.json` (filtered trigger list)

Output: `merge/train.json` (events augmented by cross-annotation; script appends and writes line by line)

## SFT Training

Based on cross-annotated outputs, construct training data and run SFT under `llama-model-sft/`.  
Reference: https://github.com/ICT-GoKnow/KnowCoder (SFT).  
When building SFT data, use cross-annotated entities/events and do not require predicting entity/event types.

## Notes

- For large datasets, run scripts from this directory to ensure relative paths resolve.
- NLTK resources are downloaded dynamically; pre-download them if running in offline environments.
- Adjust frequency thresholds and stopword lists as needed for your domains.
