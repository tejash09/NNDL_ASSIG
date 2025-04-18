# BERT-BiLSTM-CRF for Named Entity Recognition (CoNLL-2003)

This repository contains the Python code for implementing and evaluating a Named Entity Recognition (NER) system on the CoNLL-2003 dataset. The model uses a hybrid architecture combining BERT embeddings, a Bidirectional LSTM (BiLSTM) layer, and a Conditional Random Field (CRF) layer for sequence tagging.

## Project Goal

The primary objective is to build and evaluate a robust NER model capable of identifying entities like Persons (PER), Organizations (ORG), Locations (LOC), and Miscellaneous (MISC) entities within text, using the standard CoNLL-2003 benchmark dataset.

## Features

*   **Model:** BERT (`bert-base-cased`) + BiLSTM + CRF.
*   **Framework:** PyTorch.
*   **Libraries:** Leverages Hugging Face `transformers` for BERT and tokenizer, `datasets` for data loading, and `seqeval` for standard NER evaluation metrics (F1, Precision, Recall).
*   **Preprocessing:** Handles subword tokenization alignment with word-level NER tags.
*   **Training:** Includes training loop with validation, learning rate scheduling, and saving the best model based on validation F1 score.
*   **Evaluation:** Provides detailed classification reports for validation and test sets.

## Dataset

*   The code uses the **CoNLL-2003** dataset, automatically downloaded via the Hugging Face `datasets` library (`eriktks/conll2003`).
*   **Note:** Running the script for the first time will download the dataset, which may take a few minutes depending on your internet connection. The `datasets` library will cache it locally (usually in `~/.cache/huggingface/datasets/`).
*   The script uses `trust_remote_code=True` when loading the dataset as required by this specific dataset version on the Hub.

## Requirements

*   Python 3.7+
*   PyTorch (preferably with CUDA support for GPU acceleration)
*   Hugging Face `transformers`
*   Hugging Face `datasets`
*   `seqeval`
*   `numpy`
*   `tqdm`

A `requirements.txt` file is provided for easy installation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tejash09/NNDL_ASSIG.git)
    cd NNDL_ASSIG
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
