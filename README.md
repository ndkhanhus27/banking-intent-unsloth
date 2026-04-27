<div id="top"></div>

<div align="center">

# <code> Banking Intent Detection with Unsloth</code>

<em>Fine-tuning LLM for Banking Intent Classification using BANKING77 Dataset</em>

<em>Project 2 – Applications of Natural Language Processing in Industry</em>

<em>Lecturer: Dr. Nguyen Hong Buu Long</em>

<br>

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=for-the-badge&logo=tqdm&logoColor=black">
<img src="https://img.shields.io/badge/GNU%20Bash-4EAA25.svg?style=for-the-badge&logo=GNU-Bash&logoColor=white">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
<img src="https://img.shields.io/badge/Transformers-FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white">
<img src="https://img.shields.io/badge/Unsloth-000000.svg?style=for-the-badge">
<img src="https://img.shields.io/badge/LoRA-4CAF50.svg?style=for-the-badge">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=for-the-badge&logo=YAML&logoColor=white">

</div>

---

## ☀️ Table of Contents

<details>
<summary>Table of Contents</summary>

- [🌞 Overview](#-overview)
- [⚙️ Training Configuration](#️-training-configuration-unsloth-fine-tuning)
- [🔥 Features](#-features)
- [🌅 Project Structure](#-project-structure)
  - [🌄 Project Index](#-project-index)

- [🚀 Getting Started](#-getting-started)
  - [🌟 Prerequisites](#-prerequisites)
  - [⚡ Installation](#-installation)
  - [🔆 Usage](#-usage)
  - [🌠 Testing](#-testing)

- [🎥 Video Demonstration](#-video-demonstration)
- [✨ Acknowledgments](#-acknowledgments)

</details>

---

## 🌞 Overview

This project implements a **complete end-to-end pipeline for banking intent classification** using the **BANKING77 dataset** and **Unsloth framework**.

### 🎯 Objectives

- Sample a subset of BANKING77 dataset
- Perform preprocessing and data splitting
- Fine-tune a model using Unsloth
- Evaluate performance on test set
- Implement standardized inference class

---

## ⚙️ Training Configuration (Unsloth Fine-tuning)

This project strictly follows the official Unsloth fine-tuning workflow and fully documents all required hyperparameters.

---

### 🔧 Model & Framework

- Framework: **Unsloth**
- Base model: `unsloth/mistral-7b-bnb-4bit`
- Fine-tuning method: **LoRA (Low-Rank Adaptation)**

---

### 📊 Hyperparameters 

| Parameter                   | Value   |
| --------------------------- | ------- |
| Batch size                  | `4`     |
| Gradient accumulation steps | `2`     |
| Learning rate               | `2e-4`  |
| Epochs                      | `2`     |
| Max sequence length         | `256`   |
| Logging steps               | `10`    |
| Quantization                | `4-bit` |

---

### 🧠 LoRA Configuration 

| Parameter      | Value              |
| -------------- | ------------------ |
| r              | `16`               |
| target modules | `q_proj`, `v_proj` |
| lora alpha     | `16`               |
| dropout        | `0`                |

---

### 🚀 Training Pipeline

```bash
bash train.sh
```

Pipeline includes:

1. Data preprocessing (`scripts/preprocess_data.py`)
2. Model training (`scripts/train.py`)
3. Saving checkpoint

---

### 💾 Model Checkpoint

After training, model is saved to:

```bash
output_dir/
```

Includes:

- LoRA adapter
- Tokenizer
- Config files

Loaded in inference via:

```python
IntentClassification("configs/inference.yaml")
```

---

### 💡 Training Notes

- Uses **4-bit quantization** for efficiency
- Optimized for limited GPU resources
- Can run on:
  - Local GPU
  - Google Colab
  - Kaggle

---

### 🧠 Pipeline

#### 1. Data Preprocessing

- Load BANKING77 dataset
- Clean text (lowercase, remove special characters)
- Sample **100 samples per label**
- Split dataset (80/20)

#### 2. Model Training

- Unsloth + Mistral 7B
- LoRA fine-tuning

#### 3. Inference

- `IntentClassification` class
- Standardized interface

#### 4. Evaluation

- Accuracy calculation on test set

---

## 🔥 Features

|     | Component     | Details            |
| --- | ------------- | ------------------ |
| ⚙️  | Architecture  | Modular pipeline   |
| 🔩  | Model Tuning  | Unsloth + LoRA     |
| 📄  | Data Handling | Balanced BANKING77 |
| 🔌  | Configuration | YAML-based         |
| 🧩  | Inference API | Standard class     |
| 🧪  | Evaluation    | Accuracy metric    |

---

## 🌅 Project Structure

```sh
banking-intent-unsloth/
├── configs/
│   ├── inference.yaml
│   └── train.yaml
├── scripts/
│   ├── inference.py
│   ├── preprocess_data.py
│   ├── test_model.py
│   └── train.py
├── sample_data/
│   ├── train.csv
│   └── test.csv
├── train.sh
├── inference.sh
├── requirements.txt
└── README.md
```

---

## 🌄 Project Index

<details>
<summary><b>configs</b></summary>

- **train.yaml** → hyperparameters
- **inference.yaml** → model config

</details>

<details>
<summary><b>scripts</b></summary>

- **train.py** → training
- **preprocess_data.py** → data processing
- **inference.py** → prediction
- **test_model.py** → evaluation

</details>

---

## 🚀 Getting Started

### 🌟 Prerequisites

- Python 3.10+
- GPU recommended

---

### ⚡ Installation

```bash
git clone https://github.com/ndkhanhus27/banking-intent-unsloth
cd banking-intent-unsloth
pip install -r requirements.txt
```

---

### 🔆 Usage

#### Train

```bash
bash train.sh
```

#### Inference

```bash
bash inference.sh
```

or

```bash
python scripts/inference.py
```

---

### 🌠 Testing

```bash
python scripts/test_model.py
```

---

## 🎥 Video Demonstration

🎬 Demo Video:
https://drive.google.com/file/d/1MDdwB_2dGtV_xhKEmIpSG5eGnFAVMlPP/view?usp=sharing

### 📊 Final Result

- **Model Accuracy:** `0.7838`
- Evaluated on independent test set
- Demonstrates correct intent classification

---

## ✨ Acknowledgments

- Lecturer: Dr. Nguyen Hong Buu Long
- Dataset: BANKING77
- Framework: Unsloth + HuggingFace

---

<div align="right">

[Back to Top](#top)

</div>

---
