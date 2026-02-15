# Identify–Conceptualize–Align: A Schema-Adaptive Framework for Unified Entity Recognition and Event Detection

This repository contains the official implementation of:

> **Identify–Conceptualize–Align: A Schema-Adaptive Framework for Unified Entity Recognition and Event Detection**  
> Weicheng Ren, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng 
> WSDM 2026  

---

## Overview

Large Language Models (LLMs) have demonstrated strong generalization ability across tasks. However, adapting them to **Information Extraction (IE)** under *unseen schemas* remains challenging due to what we call the **schema alignment tax** — performance degradation caused by conflicts across heterogeneous schemas (e.g., synonym conflicts, label conflicts, and granularity conflicts).

To address this issue, we propose **Identify–Conceptualize–Align (ICA)**, a schema-adaptive three-phase framework for unified Named Entity Recognition (NER) and Event Detection (ED):

1. **Identification**  
   An LLM identifies entity and trigger spans in a schema-agnostic manner.  
   Cross-dataset annotation is used to improve recall and generalization.

2. **Conceptualization**  
   Each span is mapped to one or more **semantic concepts**, each consisting of:
   - `name`
   - `description`
   - `examples`  
   These concepts serve as a semantic bridge between spans and human-defined schemas.

3. **Alignment**  
   A lightweight alignment model maps concepts to schema-specific types using:
   - Transformer-based embeddings (e.g., MPNet)
   - A matching network for concept–schema classification
   - Focal loss to handle type imbalance

The first two phases are fully reusable across tasks and schemas.  
When adapting to a new schema, only the lightweight **Alignment** model needs to be retrained.

We evaluate ICA on **26 NER and ED datasets** across diverse domains.  
Experimental results demonstrate:

- +1.6% average F1 improvement in supervised settings  
- +11.5% average F1 improvement in 10-shot settings  

---

## Repository Structure

The implementation is organized into three modules:

```
> Identify/
> Conceptualize/
> Align/
```
Each corresponds to one phase of the ICA framework.

---

## Quickstart

Below is a high-level guide to reproduce the pipeline.

### 1️⃣ Identification Phase

- Build a large-scale schema-agnostic span identification dataset via cross-dataset annotation.
- Fine-tune an LLM (e.g., Qwen2.5-7B with LoRA) to perform span identification in a code-generation format.
- Output candidate entity and trigger spans.

Refer to the `Identify/` directory for data construction and training scripts.

---

### 2️⃣ Conceptualization Phase

- Convert identified spans into conceptualization prompts.
- Use an LLM (locally deployed or via API) to generate semantic concepts:
  - Concept name
  - Description
  - Examples
- Optionally generate multiple candidate concepts per span.

Refer to the `Conceptualize/` directory for preprocessing, inference server, and client scripts.

---

### 3️⃣ Alignment Phase

- Encode both conceptualized spans and schema definitions into a shared embedding space.
- Train a lightweight classification model to align concepts with schema types.
- Only this phase needs retraining when adapting to a new schema.

Refer to the `Align/` directory for training and evaluation scripts.

---

## License

This project is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

You are free to share and adapt the material, provided appropriate credit is given.

---

## Citation

If you find this work useful, please cite:

```bibtex
