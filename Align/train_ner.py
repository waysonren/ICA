import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # evaluation metrics
from torch.utils.tensorboard import SummaryWriter
import argparse
import os


class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=2, clip_prob=0.1):
        super().__init__()
        self.gamma = gamma
        self.clip_prob = clip_prob  # avoid over-focusing on extremely hard samples

    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=-1)
        ce_loss = F.nll_loss(log_softmax, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # dynamically adjust gamma: down-weight high-confidence samples
        adjusted_gamma = self.gamma * (1.0 - pt.detach()).clamp(0, 1)
        loss = (1 - pt) ** adjusted_gamma * ce_loss

        return loss.mean()

class ConceptDataset(Dataset):
    """Concept dataset with automatic text concatenation."""

    def __init__(self, candidates, concept_library, labels, sentences=None, entity_names=None):
        """
        candidates: candidate concepts per example
        concept_library: concept library (same as training)
        labels: gold label indices
        sentences: sentence list
        entity_names: entity name list
        """
        self.candidates = candidates
        self.concept_library = concept_library
        self.labels = labels
        self.sentences = sentences if sentences is not None else [""] * len(candidates)
        self.entity_names = entity_names if entity_names is not None else [""] * len(candidates)

    def __len__(self):
        return len(self.candidates)

    def _concat_text(self, concept, sentence="", entity_name=""):
        return f"Concept: {concept['name']} [SEP] Description: {concept['description']} [SEP] Example: {concept['example']} [SEP] Sentence: {sentence} [SEP] Entity: {entity_name}"

    def __getitem__(self, idx):
        candidates = []
        for concept in self.candidates[idx]:
            candidates.append(self._concat_text(
                concept,
                self.sentences[idx],
                self.entity_names[idx]
            ))
        return candidates, self.labels[idx]


def collate_fn(batch):
    # keep a sample-centric structure
    return [item[0] for item in batch], torch.tensor([item[1] for item in batch])


class NeuralConceptModel(nn.Module):
    """Interactive model with a matching network."""

    def __init__(self, concept_texts,
                 model_name='./all-mpnet-base-v2',
                 hidden_dim=512,
                 num_layers=2,
                 num_classes=6,
                 tune_layers=3):
        super().__init__()
        self.concept_texts = concept_texts  # store all concept texts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)  # ensure the model is on the correct device

        # infer encoder hidden size dynamically
        test_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
        test_input = {k: v.to(device) for k, v in test_input.items()}  # move inputs to device
        with torch.no_grad():
            test_output = self.encoder(**test_input)
        encoder_dim = test_output.last_hidden_state.size(-1)

        # set matching network dimensions
        self.match_net = nn.Sequential(
            nn.Linear(encoder_dim * 3, hidden_dim),
            nn.GELU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU()
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, num_classes)  # output shape: [batch_size, num_classes]
        ).to(device)
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # unfreeze the last tune_layers layers
        try:
            total_layers = len(self.encoder.encoder.layer)
            for i in range(total_layers - tune_layers, total_layers):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
        except AttributeError:
            print("WARNING: encoder.encoder.layer not found. Check your model structure.")

    def _concat_features(self, emb1, emb2):
        """Ensure the input tensor dimensions match."""
        if emb1.dim() == 2 and emb2.dim() == 2:
            if emb1.size(0) != emb2.size(0):  # if batch sizes do not match
                emb2 = emb2.expand(emb1.size(0), -1)  # expand emb2 to match emb1 batch size

        diff = torch.abs(emb1 - emb2)
        return torch.cat([emb1, emb2, diff], dim=-1)

    def forward(self, candidate_texts):
        # encode candidate texts
        # batch_candidate_texts = list(map(list, zip(*candidate_texts)))
        # candidate_emb = self.encode_texts_list(batch_candidate_texts)  # [batch_size, emb_dim]
        candidate_emb = self.encode_texts_list(candidate_texts)
        # encode all concept texts
        concept_embs = self.encode_texts(self.concept_texts)  # [num_concepts, emb_dim]

        # compute matching scores between each candidate and all concepts
        logits = []
        for concept_emb in concept_embs:
            concept_emb = concept_emb.unsqueeze(0).expand(candidate_emb.size(0), -1)  # [batch_size, emb_dim]
            combined = self._concat_features(candidate_emb, concept_emb)  # [batch_size, emb_dim * 3]
            score = self.match_net(combined)  # [batch_size, num_class]
            logits.append(score)
        logits = torch.stack(logits, dim=1)  # [batch_size, num_concepts, num_class]

        # take the max score over concepts for each candidate
        logits = logits.max(dim=1).values  # [batch_size, num_class]
        return logits

    def encode_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = self.encoder(**inputs)
        return self._mean_pooling(outputs, inputs['attention_mask'])

    def encode_texts_list(self, texts_list):
        """
        Efficiently batch-encode multiple concept texts per entity and aggregate at the entity level.
        - texts_list: List[List[str]]
        - return: Tensor [batch_size, hidden_dim]
        """
        # flatten all concept texts
        flat_texts = [text for sublist in texts_list for text in sublist]

        inputs = self.tokenizer(
            flat_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        outputs = self.encoder(**inputs)
        all_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])  # [batch_size, emb_dim]

        # aggregate per entity (max pooling)
        final_embeddings = []
        idx = 0
        for concepts in texts_list:
            concept_embs = all_embeddings[idx: idx + len(concepts)]  # [n_concepts_i, emb_dim]
            entity_emb = torch.max(concept_embs, dim=0).values  # [emb_dim]
            final_embeddings.append(entity_emb)
            idx += len(concepts)

        return torch.stack(final_embeddings)  # [batch_size, emb_dim]

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# evaluation
def evaluate(model, test_loader, original_data_file):
    # build inverse mapping from id to label
    id2label = {v: k for k, v in label2id.items()}

    # read original data
    with open(original_data_file, 'r') as f:
        data_lines = [json.loads(line) for line in f]
    model.eval()
    all_preds, all_labels = [], []
    pred_idx = 0  # global prediction index

    # first compute how many entities need prediction
    total_entities = 0
    for line in data_lines:
        if "pred_entities_concept" in line:
            total_entities += len(line["pred_entities_concept"])
    # ensure test set size matches the JSON file
    assert len(test_loader.dataset) == total_entities, \
        f"Data mismatch: test set has {len(test_loader.dataset)} items, but JSON file has {total_entities} items"

    with torch.no_grad():
        total_test_batches = len(test_loader)

        update_interval = max(1, int(total_test_batches * 0.2))  # update every 20%
        # use dynamic update settings
        progress_bar = tqdm(
            test_loader,
            total=total_test_batches,  # total is required to estimate remaining time
            mininterval=float('inf'),  # fully disable time-triggered updates
            miniters=update_interval,
        )

        for batch_idx, (texts, labels) in enumerate(progress_bar):
            if (batch_idx + 1) % update_interval == 0:
                progress_bar.refresh()
            logits = model(texts)  # [batch_size, num_class]
            preds = logits.argmax(dim=1).cpu().numpy()  # [batch_size]
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            # write predictions back into the original data structure
            for pred in preds:
                # locate the line and entity corresponding to this prediction
                line_idx = 0
                entity_idx = 0
                found = False

                # search which line/entity this prediction belongs to
                while line_idx < len(data_lines) and not found:
                    if "pred_entities_concept" in data_lines[line_idx]:
                        if pred_idx - entity_idx < len(data_lines[line_idx]["pred_entities_concept"]):
                            # found the target slot for this prediction
                            data_lines[line_idx]["pred_entities_concept"][pred_idx - entity_idx]["pred_type"] = \
                                id2label[pred]
                            found = True
                        else:
                            entity_idx += len(data_lines[line_idx]["pred_entities_concept"])
                            line_idx += 1
                    else:
                        line_idx += 1

                if not found:
                    raise IndexError(f"Cannot find a matching entity for prediction index {pred_idx}")

                pred_idx += 1
    # compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    classification_report_result = classification_report(all_labels, all_preds, digits=4)
    print(classification_report_result)
    print("Confusion Matrix:\n", cm)
    tp = sum(np.diag(cm)[:-1])  # True Positives: diagonal elements
    predict_number = sum(np.sum(cm, axis=0)[:-1])  # predicted per-class count (excluding Other)
    return accuracy, tp, predict_number, data_lines, classification_report_result, cm





def save_model(model, save_dir, epoch, optimizer, loss):
    # save full training state
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"{save_dir}/checkpoint.pt")
    print(f"Model saved to {save_dir}")


def read_data(annotation_file, only_first_concept=False):
    with open(annotation_file, "r") as f:
        candidates = []
        labels = []
        sentences = []
        entity_names = []
        golden_ent_number = 0
        for raw_line in f:
            line = json.loads(raw_line)
            if "pred_entities_concept" in line.keys():
                concept_key = "pred_entities_concept"
            else:
                concept_key = "entities"
            sentence = line["sentence"]

            for ent in line[concept_key]:
                if "concept" in ent.keys():
                    entity_name = ent["name"]
                    if only_first_concept:
                        candidates.append([ent["concept"][0]])
                        labels.append(label2id[ent["type"]])
                        sentences.append(sentence)
                        entity_names.append(entity_name)
                    else:
                        candidates.append(ent["concept"])
                        labels.append(label2id[ent["type"]])
                        sentences.append(sentence)
                        entity_names.append(entity_name)
            golden_ent = [ent for ent in line["entities"] if ent["type"] != "Other"]
            golden_ent_number += len(golden_ent)
    return candidates, labels, sentences, entity_names, golden_ent_number


def process_schema(schema_file):
    with open(schema_file, "r") as f:
        concept_library = json.load(f)
    label2id = {}
    id_cnt = 0
    for concept in concept_library:
        label2id[concept['name']] = id_cnt
        id_cnt += 1
    return concept_library, label2id


# training loop
def train_classifier(model, train_loader, test_loader, optimizer, epochs, gold_mention_n, annotation_file_test):
    focal_loss = AdaptiveFocalLoss()
    best_score = 0.0
    meta = {}
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        update_interval = max(1, int(total_batches * 0.2))  # update every 20%

        # use dynamic update settings
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            total=total_batches,  # total is required to estimate remaining time
            mininterval=float('inf'),     # fully disable time-triggered updates
            miniters=update_interval,
        )

        for batch_idx, batch in enumerate(progress_bar):
            texts, labels = batch
            labels = labels.to(device)

            # forward
            logits = model(texts)
            loss = focal_loss(logits, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            total_loss += loss.item()

            # current average loss (batch_idx + 1 is more accurate)
            current_loss = total_loss / (batch_idx + 1)

            # force display updates (every 20%)
            if (batch_idx + 1) % update_interval == 0:
                precent = 0.2 * ((batch_idx + 1) / update_interval)
                progress_bar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "processed": f"{precent:.1f}",
                    "lr_bert": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "lr_match_net": f"{optimizer.param_groups[1]['lr']:.2e}"
                })
                progress_bar.refresh()  # ensure display refresh

            writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_loader) + batch_idx)
        if epoch < 10:
             continue
        # evaluate once per epoch
        model.eval()
        library_texts = [test_dataset._concat_text(c) for c in concept_library]
        concept_embeddings = model.encode_texts(library_texts).cpu().detach().numpy()
        test_accuracy, true_positive_n, pred_mention_n, data_lines, classification_report, cm = evaluate(model, test_loader, annotation_file_test)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        prec_c = true_positive_n / pred_mention_n if pred_mention_n != 0 else 0
        recall_c = true_positive_n / gold_mention_n if gold_mention_n != 0 else 0
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
        print("{:.4f}, {:.4f}, {:.4f},".format(prec_c, recall_c, f1_c), true_positive_n, gold_mention_n, pred_mention_n)

        # save model
        # save_model(model, epoch_save_dir)

        # save best model
        if f1_c > best_score:
            # save training metadata
            meta = {
                "epoch": epoch + 1,
                "loss": total_loss / len(train_loader),
                "save_time": datetime.now().isoformat(),
                "model_type": type(model.encoder).__name__,
                "test_accuracy_include_other": test_accuracy,
                "f1_c_no_other": f1_c,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "classification_report": classification_report,
                "cm": cm.tolist()
            }
            with open(f"{BASE_SAVE_PATH}/metadata.json", "w") as f:
                json.dump(meta, f)
            with open(f"{BASE_SAVE_PATH}/predict.json", 'w') as f:
                for line in data_lines:
                    f.write(json.dumps(line) + '\n')
            best_score = f1_c
            save_model(model, save_dir=f"{BASE_SAVE_PATH}/best_model", epoch=EPOCHS, optimizer=optimizer,
                       loss=total_loss / len(train_loader))

            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {meta['loss']:.4f} | Model saved to {BASE_SAVE_PATH}")

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / len(train_loader):.4f}")
    print(meta)
    print(f"Model saved to {BASE_SAVE_PATH}")



# training loop
def eval_classifier(model, train_loader, test_loader, optimizer, epochs, gold_mention_n, annotation_file_test, best_model_path):
    for epoch in range(EPOCHS):
        model.eval()
        test_accuracy, true_positive_n, pred_mention_n, data_lines, classification_report, cm = evaluate(model, test_loader, annotation_file_test)

        prec_c = true_positive_n / pred_mention_n if pred_mention_n != 0 else 0
        recall_c = true_positive_n / gold_mention_n if gold_mention_n != 0 else 0
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c) if prec_c or recall_c else 0
        print("{:.4f}, {:.4f}, {:.4f},".format(prec_c, recall_c, f1_c), true_positive_n, gold_mention_n, pred_mention_n)

        with open(f"{best_model_path}/predict.json", 'w') as f:
            for line in data_lines:
                f.write(json.dumps(line) + '\n')


def load_model(save_dir, num_layers, hidden_dim, model_name, concept_texts, concept_library, tune_layers):
    # load full training state
    model = NeuralConceptModel(num_layers=num_layers, hidden_dim=hidden_dim, concept_texts=concept_texts,
                               model_name=model_name, num_classes=len(concept_library), tune_layers=tune_layers).to(device)
    # print("Current model structure:")
    # print(model)
    checkpoint = torch.load(f"{save_dir}/checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)

# root save directory with a timestamp
BASE_SAVE_PATH = f"./saved_models/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
Path(BASE_SAVE_PATH).mkdir(parents=True, exist_ok=True)
Path(BASE_SAVE_PATH + "/best_model").mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=BASE_SAVE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="NER")
parser.add_argument('--file_dir', type=str, default="ACE2005_first_concept")
parser.add_argument('--base_dir', type=str, default="./data")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--model_name', type=str, default="./all-MiniLM-L6-v2")
parser.add_argument('--continue_train', type=bool, default=False)
parser.add_argument('--best_model', type=str, default="./all-MiniLM-L6-v2")
parser.add_argument('--predict', type=bool, default=False)
parser.add_argument('--tune_layers', type=int, default=3, help="Number of last MPNet encoder layers to fine-tune")
parser.add_argument('--batch_size', type=int, default=8, help="batch size")


args = parser.parse_args()

task = args.task
file_dir = args.file_dir
base_dir = args.base_dir
device = args.device

# training configuration
num_layers = args.num_layers
hidden_size = args.hidden_size
EPOCHS = args.epochs
model_name = args.model_name
BATCH_SIZE = args.batch_size
LR = 2e-5

os.environ["CUDA_VISIBLE_DEVICES"] = device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(model_name)

schema_file = f"{base_dir}/{task}/{file_dir}/schema.json"
concept_library, label2id = process_schema(schema_file)

annotation_file_train = f"{base_dir}/{task}/{file_dir}/train.jsonl"
candidates, labels, sentences, entity_names, gold_mention_n_train = read_data(annotation_file_train, only_first_concept=False)
train_dataset = ConceptDataset(candidates, concept_library, labels, sentences, entity_names)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

annotation_file_test = f"{base_dir}/{task}/{file_dir}/test.jsonl"
candidates, labels, sentences, entity_names, gold_mention_n = read_data(annotation_file_test, only_first_concept=False)
test_dataset = ConceptDataset(candidates, concept_library, labels, sentences, entity_names)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(annotation_file_train)
print(annotation_file_test)

# build concept texts
concept_texts = [test_dataset._concat_text(c) for c in concept_library]
# initialize model and optimizer
# model = NeuralConceptModel().to(device)

if args.continue_train:
    best_model_path = args.best_model
    model_dir = f"{best_model_path}/best_model"
    model = load_model(model_dir, num_layers, hidden_size, model_name, concept_texts, concept_library, args.tune_layers)
else:
    model = NeuralConceptModel(num_layers=num_layers, hidden_dim=hidden_size, concept_texts=concept_texts,
                               model_name=model_name, num_classes=len(concept_library), tune_layers=args.tune_layers).to(device)

optimizer = torch.optim.AdamW([
    {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 'lr': 1e-5},
    {'params': model.match_net.parameters(), 'lr': 1e-4}
], weight_decay=0.01)


if args.predict:
    eval_classifier(model, train_loader, test_loader, optimizer, EPOCHS, gold_mention_n, annotation_file_test, args.best_model)
else:
    train_classifier(model, train_loader, test_loader, optimizer, EPOCHS, gold_mention_n, annotation_file_test)
