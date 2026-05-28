# 🚗 Prototype Car Review LLM Chatbot

![car](car.jpeg)

The CTO at **"Car-ing is sharing"**, a car sales and rental company, requested a prototype for a chatbot application designed to address diverse customer inquiries and assist human support agents. Processing raw customer reviews manually is slow and inefficient; this prototype introduces an automated system to interpret, translate, query, and distill automotive text data at scale.

This project implements a multi-task **Natural Language Processing (NLP)** pipeline utilizing fine-tuned, pre-trained **Large Language Models (LLMs)** hosted on the Hugging Face Hub to perform automated classification, translation, question-answering, and text summarization.

---

## 📌 Guiding Questions
- How effectively can a lightweight, fine-tuned transformer (`DistilBERT`) classify customer sentiments compared to ground-truth text data?
- Can a specialized machine translation model (`MarianMT`) maintain contextual integrity and score highly on BLEU evaluation metrics?
- How accurately can an extractive QA model locate explicit text details answering customer inquiries about brand preferences?
- What are the operational token boundaries required to generate concise summaries without losing the core intent of a long review?

---

## 📂 The Data

The pipeline evaluates customer sentiment and feedback using a multi-column dataset (`car_reviews.csv`) alongside a validation reference sheet:

### `car_reviews.csv`
| Column | Data Type | Description |
|--------|-----------|-------------|
| `Review` | `VARCHAR` | The raw text feedback detailing a customer's experience with a rented or purchased vehicle. |
| `Class` | `VARCHAR` | Ground-truth text sentiment labels classifying the feedback as either `POSITIVE` or `NEGATIVE`. |

### `reference_translations.txt`
Contains high-quality human-translated Spanish reference strings for validation matching of English-to-Spanish translation accuracy scores.

---

## 🛠️ Project Steps

### 1. Sentiment Classification & Evaluation
- Ingested the dataset and passed the target feedback items through a specialized `distilbert-base-uncased-finetuned-sst-2-english` pipeline.
- Extracted raw dictionary prediction labels into `predicted_labels` and mapped categorical keys to integer binary arrays `{0, 1}`.
- Deployed the `evaluate` metrics engine to compute scalar floating-point representations for both **Accuracy** and **F1-Score**.

### 2. English-to-Spanish Translation
- Targeted reviews from emerging Spanish markets and isolated feedback data for translation.
- Deployed the `Helsinki-NLP/opus-mt-en-es` model to translate text data, using sequence parameter constraints (`max_length=27`) to capture the first two sentences naturally.
- Cleaned human target outputs from `reference_translations.txt` and evaluated model generations by computing an overall **BLEU Score** dictionary.

### 3. Extractive Question Answering
- Implemented an extractive QA architecture utilizing `deepset/minilm-uncased-squad2`.
- Formulated a target business prompt: *"What did he like about the brand?"*
- Fed the structured prompt and the review context into the model to isolate and extract the precise semantic string answer directly from the text block.

### 4. Text Summarization
- Handled long-form document feedback by loading a heavy-duty sequence-to-sequence model (`facebook/bart-large-cnn`).
- Implemented explicit token bounds (`min_length=50`, `max_length=55`) alongside deterministic decoding (`do_sample=False`) to synthesize a dense, 50-55 token summary stored inside `summarized_text`.

---

## ✅ Key Deliverables
- **Multi-Task NLP Notebook**: A complete, functional workspace testing diverse transformer architectures for production readiness.
- **Automated Performance Tracking**: Clean evaluation workflows tracking accuracy floats and BLEU score dictionaries to ensure the models meet corporate precision criteria.

---

## 🧰 Skills Used
- Python
- Hugging Face Ecosystem (`transformers`, `pipeline`)
- Model Evaluation Standards (`evaluate`)
- Large Language Models (LLMs) & Tokenization
- Text Processing & Hyperparameter Slicing (`max_length`)
