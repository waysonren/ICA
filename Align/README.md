# Align (ED / NER)

This folder contains two training scripts for **concept-based classification**:

- `train_ed.py`: event detection (ED) type classification
- `train_ner.py`: named entity recognition (NER) type classification

Both scripts use a Transformer encoder (via `transformers`) plus a lightweight matching network. Candidate concept texts are built from concept name/description/example, and aggregated at the entity/event level via max pooling.

## Environment

Python dependencies are pinned in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:

- `torch`
- `transformers`
- `scikit-learn`
- `tensorboard`
- `tqdm`

## Data preparation

Both scripts expect the following directory layout under `--base_dir`:

```
<base_dir>/
  <task>/                # can include subfolders like "ED_structure_v4_low_resource2/20"
    <file_dir>/
      schema.json
      train.jsonl
      test.jsonl
```

### `schema.json`

A JSON list of concepts. Each concept is a dict containing:

- `name`: label string (used to build `label2id`)
- `description`: text description
- `example`: example text

### `train.jsonl` / `test.jsonl`

Each line is a JSON object with at least:

- `sentence`: the original sentence text

And one of the following fields (depending on the script and stage):

- For ED (`train_ed.py`)
  - training/eval: `events`
  - prediction/eval on predicted candidates: `pred_events_concept`
  - each event item should contain:
    - `type`: the gold label (must match a `schema.json` concept `name`, plus optionally `Other`)
    - `trigger`: event trigger string (used as entity name input)
    - `concept`: a list of candidate concept objects (each concept object includes `name`, `description`, `example`)

- For NER (`train_ner.py`)
  - training/eval: `entities`
  - prediction/eval on predicted candidates: `pred_entities_concept`
  - each entity item should contain:
    - `type`: the gold label (must match a `schema.json` concept `name`, plus optionally `Other`)
    - `name`: entity mention string (used as entity name input)
    - `concept`: a list of candidate concept objects (each concept object includes `name`, `description`, `example`)

## Run commands

### Train ED

```bash
python -u train_ed.py \
  --task taks_name(NER/ED) \
  --file_dir /path/to/data \
  --base_dir /path/to/data \
  --device 0 \
  --epochs 30 \
  --num_layers 3 \
  --hidden_size 128 \
  --model_name /path/to/all-mpnet-base-v2 \
  --tune_layers 12 \
  --batch_size 8
```

Artifacts are written under:

```
./saved_models/<timestamp>/
  metadata.json
  predict.json
  best_model/checkpoint.pt
```

Note: evaluation and best-model saving start after epoch 10 (`epoch < 10` is skipped).

### Train NER

```bash
python -u train_ner.py \
  --task taks_name(NER/ED) \
  --file_dir dataset_name \
  --base_dir /path/to/data \
  --device 0 \
  --epochs 30 \
  --num_layers 3 \
  --hidden_size 128 \
  --model_name /path/to/all-mpnet-base-v2 \
  --tune_layers 12 \
  --batch_size 8
```

### Predict / evaluate with a saved checkpoint

Both scripts support a simple evaluation-only mode that loads `checkpoint.pt` from `<best_model_path>/best_model/`:

```bash
python -u train_ed.py \
  --task taks_name(NER/ED) \
  --file_dir dataset_name \
  --base_dir /path/to/data \
  --device 0 \
  --model_name /path/to/all-mpnet-base-v2 \
  --continue_train True \
  --predict True \
  --best_model ./saved_models/<timestamp>
```

Output is written to:

- ED: `<best_model_path>/predict2.json`
- NER: `<best_model_path>/predict.json`

Reminder: `argparse` uses `type=bool` here, so pass explicit `True/False` values (e.g., `--predict True`).

### TensorBoard

```bash
tensorboard --logdir ./saved_models
```
