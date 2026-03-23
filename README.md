# Resume Analyzer
### Automated Resume Analysis using DistilBERT + T5 + spaCy NER + SBERT

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square)
![spaCy](https://img.shields.io/badge/spaCy-NER-09a3d5?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-99%25-brightgreen?style=flat-square)

---

## Overview

A multi-model NLP system that analyzes resumes in real time. Paste any resume and instantly get a professional summary, predicted job category, extracted entities, and a job description match score — all powered by four different AI models working together.

Built as an end-to-end NLP project covering data preprocessing, model fine-tuning, evaluation, and Streamlit deployment.

---

## Demo

```
Input: Python Developer resume with TensorFlow, AWS, Django experience

Summary (T5):          "Experienced Python developer with ML pipeline expertise..."
Category (DistilBERT): Python Developer — 91.3% confidence
Skills (spaCy NER):    Python, TensorFlow, PyTorch, AWS, Docker, Django, Flask
Job Match (SBERT):     78% — Good match for Data Engineer role
```

---

## Four models — four different tasks

| Model | Task | Architecture | Parameters |
|-------|------|-------------|-----------|
| DistilBERT | Resume category classification | Encoder-only transformer | 66M |
| T5-small | Resume text summarization | Encoder-Decoder transformer | 60M |
| spaCy en_core_web_sm | Named entity recognition | CNN + rule-based | 12M |
| SBERT all-MiniLM-L6-v2 | Job description match score | Siamese BERT | 22M |

This combination covers four different NLP paradigms in one project — classification, generation, token labeling, and semantic similarity.

---

## Project Structure

```
my_project/
│
├── app.py                      ← Streamlit web application
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
└── resume_classifier/          ← Fine-tuned DistilBERT (download separately)
    ├── model.safetensors       ← Fine-tuned weights
    ├── config.json             ← Model architecture config
    ├── tokenizer.json          ← DistilBERT vocabulary
    ├── tokenizer_config.json   ← Tokenizer settings
    ├── vocab.txt               ← Token vocabulary
    ├── label_map.json          ← Category name to integer mapping
    ├── id2label.json           ← Integer to category name mapping
    └── metrics.json            ← Accuracy, F1, confusion matrix data
```

T5, spaCy, and SBERT load automatically from HuggingFace on first run — no manual download needed for these three.

---

## Streamlit App — Three Tabs

**Tab 1 — Resume Analyzer**
- Paste any resume text
- AI-generated professional summary (T5)
- Predicted job category with confidence score (DistilBERT)
- Top 5 category predictions with probability bars
- Extracted skills, organisations, locations, dates (spaCy NER)

**Tab 2 — Job Match**
- Paste resume and job description side by side
- Cosine similarity match score (SBERT)
- Color-coded result — Excellent / Good / Partial / Low match
- Actionable tips to improve match score

**Tab 3 — Model Performance**
- Overall accuracy, macro F1, precision, recall
- Per-class performance table for all 25 categories
- Model architecture details and training configuration

---

## Dataset

**Resume Dataset** — kaggle.com/datasets/gauravduttakiit/resume-dataset

| Property | Value |
|----------|-------|
| Total resumes | 2,484 |
| Job categories | 25 |
| Format | CSV with Resume and Category columns |

### Categories covered

Java Developer, Python Developer, Data Science, Web Designing, HR, Advocate,
Arts, Automation Testing, Blockchain, Business Analyst, Civil Engineer,
Database, DevOps Engineer, DotNet Developer, Electrical Engineering,
ETL Developer, Hadoop, Health and Fitness, Mechanical Engineer,
Network Security Engineer, Operations Manager, PMO, SAP Developer, Sales, Testing

---

## Training Details — DistilBERT Classifier

```
Base model:      distilbert-base-uncased
Task:            25-class resume classification
Epochs:          10 with early stopping (patience=2)
Batch size:      8 train / 16 eval
Optimizer:       AdamW
Learning rate:   2e-5
LR scheduler:    Linear warmup 10% + linear decay
Weight decay:    0.01
Max token len:   256
fp16:            False — avoids gradient underflow
Hardware:        Google Colab T4 GPU
Training time:   ~8 minutes
```

### Why DistilBERT not RoBERTa?

DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its accuracy. For a 25-class resume classifier where categories have distinctive vocabulary, this trade-off is worthwhile. RoBERTa would give marginally higher accuracy but triple the inference time on CPU deployment.

### Why T5 for summarization?

T5 frames every NLP task as text generation. Input: `"summarize: [resume text]"` → Output: professional summary. No classification head needed — the encoder-decoder architecture handles generation naturally.

### Why SBERT for job matching?

Standard BERT embeddings are not directly comparable across sentences. SBERT fine-tunes BERT with a siamese network so sentence embeddings can be compared using cosine similarity. `all-MiniLM-L6-v2` is CPU-friendly at 80MB with excellent similarity scores.

---

## Real Problems Solved During Development

**Synthetic data problem** — Initial dataset had 200,000 rows but only 10 unique descriptions repeated 20,000 times each. This produced fake 100% accuracy through memorisation. Switching to genuinely unique data gave honest results.

**fp16 gradient underflow** — Enabling mixed precision caused all gradients to underflow to zero on certain GPU instances, producing 20% accuracy (random guessing) across all epochs. Disabling fp16 fixed it immediately.

**Encoding corruption** — Resume text contained non-ASCII characters from PDF copy-paste. Fixed with ASCII encoding before tokenization.

**Pipeline version mismatch** — HuggingFace v5.x removed summarization pipeline task names. Solved by loading T5 directly without the pipeline abstraction.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/resume-analyzer.git
cd resume-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy English model

```bash
python -m spacy download en_core_web_sm
```

### 4. Download the trained DistilBERT classifier

Download `resume_classifier.zip` from the [Releases](../../releases) page and unzip:

```bash
unzip resume_classifier.zip -d resume_classifier/
```

### 5. Run the app

```bash
# Standard
streamlit run app.py

# Windows if streamlit not recognized
python -m streamlit run app.py
```

Open browser at `http://localhost:8501`

> First load takes 2-3 minutes as T5 and SBERT download from HuggingFace automatically. After first load everything is cached and instant.

---

## Training Your Own Model

1. Download `UpdatedResumeDataSet.csv` from Kaggle
2. Open `ResumeAnalyzer.ipynb` in Google Colab
3. Set runtime to T4 GPU — Runtime → Change runtime type → T4 GPU
4. Run all cells in order
5. Download `resume_classifier.zip` from the final cell

---

## Requirements

```
streamlit
transformers
torch
spacy
sentence-transformers
numpy
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Classification | DistilBERT (distilbert-base-uncased) |
| Summarization | T5-small |
| NER | spaCy en_core_web_sm |
| Similarity | SBERT all-MiniLM-L6-v2 |
| Training framework | PyTorch + HuggingFace Transformers |
| Training platform | Google Colab T4 GPU |
| Web interface | Streamlit |
| Language | Python 3.11+ |

---

## Key Learnings

**Four architectures in one project** — Encoder-only (DistilBERT), Encoder-Decoder (T5), CNN+rules (spaCy), and Siamese networks (SBERT) each solve a fundamentally different NLP problem.

**Fine-tuning vs pretrained zero-shot** — DistilBERT was fine-tuned on resume data. T5, spaCy, and SBERT were used pretrained — showing when to fine-tune vs when pretrained weights are sufficient.

**Warmup scheduler is essential** — Starting with a tiny learning rate prevents the randomly-initialised classification head from corrupting pretrained weights in early training steps.

**Data quality beats quantity** — 2,484 unique real resumes produced better results than 200,000 synthetic duplicates. Quality always wins in NLP fine-tuning.

---

## References

- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
- Raffel et al. (2020). *Exploring the Limits of Transfer Learning with T5.* JMLR.
- Reimers & Gurevych (2019). *Sentence-BERT.* EMNLP 2019.
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization.* ICLR.
- Wolf et al. (2020). *Transformers: State-of-the-Art NLP.* HuggingFace. EMNLP.

---

## License

Academic project. Dataset sourced from Kaggle under public license.
Model weights subject to HuggingFace and respective model license terms.

---

*Built as part of an NLP academic project — March 2026*
