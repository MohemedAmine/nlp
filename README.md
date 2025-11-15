# Natural Language Processing Labs

## Complete Practical Training Series

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-green?style=flat-square)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey?style=flat-square)](./LICENSE)

**A comprehensive practical training series covering fundamental to advanced NLP techniques**

[Overview](#overview) • [Labs](#-labs) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Learning Path](#-learning-path)

</div>

---

## Overview

This repository provides a **structured learning experience** in Natural Language Processing, progressing from basic text preprocessing through state-of-the-art deep learning architectures. Each lab (TP1-TP5) builds upon previous concepts, offering hands-on practice with industry-standard libraries and techniques.

**Target Audience:** Students and practitioners learning NLP fundamentals to advanced concepts.

---

## 🎓 Labs Overview

### **Lab 1: Text Preprocessing & Normalization**

| **Objective** | Foundational text cleaning and normalization techniques |
| :------------ | :------------------------------------------------------ |
| **File**      | `tp1/tp1.ipynb`                                         |
| **Dataset**   | `spooky.csv`                                            |
| **Domain**    | Author Classification                                   |

**Core Concepts:**

- Text loading and exploratory data analysis
- Character normalization (accent removal via Unicode NFKD)
- Duplicate character reduction with regex patterns
- URL and email detection/replacement
- Data validation and quality assessment
- Statistical visualization of text distributions

**Skills Acquired:** Data inspection, regex patterns, text normalization pipeline

---

### **Lab 2: Feature Extraction & Classification**

| **Objective** | Converting text to numerical features for machine learning |
| :------------ | :--------------------------------------------------------- |
| **File**      | `tp2/tp2.ipynb`                                            |
| **Dataset**   | `spooky_cleaned.csv`                                       |
| **Domain**    | Multi-class Author Attribution                             |

**Core Concepts:**

- **Vectorization Methods:**
  - Bag of Words (CountVectorizer) - frequency-based features
  - TF-IDF (TfidfVectorizer) - weighted importance scores
- Dimensionality reduction via t-SNE visualization
- Stratified train-test splitting for balanced evaluation
- Label encoding and categorical transformation
- Neural network architecture design with Keras/TensorFlow
- Model training, validation, and performance metrics

**Skills Acquired:** Feature engineering, model development, deep learning fundamentals

---

### **Lab 3: Advanced Feature Engineering & Validation**

| **Objective** | Sophisticated feature design with robust validation methodology |
| :------------ | :-------------------------------------------------------------- |
| **File**      | `tp3/tp3.ipynb`                                                 |
| **Dataset**   | `spooky_cleaned.csv`                                            |
| **Domain**    | Robust Classification Pipeline                                  |

**Core Concepts:**

- One-hot encoding for categorical variables
- Stratified k-fold cross-validation for reliable evaluation
- **Advanced TF-IDF:** Feature reduction with `max_features`
- N-gram modeling (unigrams, bigrams, trigrams)
- Class imbalance handling and mitigation strategies
- Feature importance ranking and selection
- Statistical validation techniques

**Skills Acquired:** Cross-validation, feature selection, imbalanced data handling

---

### **Lab 4: Sequence Labeling & Named Entity Recognition**

| **Objective** | Token-level classification using sequence models           |
| :------------ | :--------------------------------------------------------- |
| **File**      | `tp4/tp4.ipynb`                                            |
| **Datasets**  | `conllpp_train.txt`, `conllpp_test.txt`, `conllpp_dev.txt` |
| **Domain**    | Named Entity Recognition (NER)                             |

**Core Concepts:**

- **Embedding Layers:** GloVe (Global Vectors for Word Representation)
- **Sequence Models:**
  - BiLSTM (Bidirectional LSTM) for contextual information capture
  - GRU (Gated Recurrent Units) - efficient RNN variant
  - 1D CNN for feature extraction
- Sequence padding and dynamic length handling
- CoNLL++ dataset format and parsing
- Token-level classification and loss functions
- F1-score evaluation for sequence labeling tasks
- Early stopping and regularization techniques

**Skills Acquired:** RNN architectures, embeddings, sequence modeling, NER evaluation

---

### **Lab 5: Neural Machine Translation with Transformers**

| **Objective** | Sequence-to-sequence translation using attention mechanisms |
| :------------ | :---------------------------------------------------------- |
| **File**      | `tp5/Neural_Machine_Translation_(Transformer).ipynb`        |
| **Dataset**   | `en-fr.txt` (English-French parallel corpus)                |
| **Domain**    | Machine Translation (EN↔FR)                                 |

**Core Concepts:**

- **Transformer Architecture:**
  - Self-attention mechanism with scaled dot-product
  - Multi-head attention for diverse feature representation
  - Positional encoding for sequence ordering
  - Feed-forward networks and layer normalization
- Encoder-Decoder framework
- Sequence-to-sequence (Seq2Seq) learning
- Beam search for inference optimization
- BLEU score for translation quality evaluation
- Model training with gradient accumulation
- Inference and decoding strategies

**Skills Acquired:** Transformers, attention mechanisms, seq2seq models, translation evaluation

---

### **TP1: Text Preprocessing & Cleaning**

**File:** `tp1/tp1.ipynb`

Text preprocessing fundamentals using the Spooky dataset (author classification).

**Key Topics:**

- Loading and exploring text data with pandas
- Removing repeating characters (e.g., "coool" → "cool")
- Removing accented characters with Unicode normalization
- URL and email removal/replacement
- Text cleaning pipeline
- Data visualization and distribution analysis

**Dataset:** `spooky.csv`

---

### **TP2: Feature Extraction & Text Classification with Deep Learning**

**File:** `tp2/tp2.ipynb`

Feature extraction techniques and neural network-based text classification.

**Key Topics:**

- Feature extraction methods:
  - Bag of Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
- Dimensionality reduction with t-SNE
- Train-test split with stratification
- Label encoding for multi-class classification
- Building and training neural networks using Keras/TensorFlow
- Model evaluation and visualization

**Dataset:** `spooky_cleaned.csv` (preprocessed from TP1)

---

### **TP3: Advanced Feature Engineering & Cross-Validation**

**File:** `tp3/tp3.ipynb`

Advanced feature engineering techniques with stratified k-fold cross-validation.

**Key Topics:**

- One-hot encoding for categorical features
- Stratified k-fold cross-validation for balanced data split
- TF-IDF with feature reduction:
  - `max_features` parameter for dimensionality control
  - N-gram support (bigrams, trigrams)
- Feature importance analysis
- Handling imbalanced datasets

**Dataset:** `spooky_cleaned.csv`

---

### **TP4: Named Entity Recognition (NER) with BiLSTM**

**File:** `tp4/tp4.ipynb`

Sequence labeling for Named Entity Recognition using recurrent neural networks.

**Key Topics:**

- Loading CoNLL++ dataset (sequence labeling format)
- Word embeddings (GloVe - Global Vectors for Word Representation)
- BiLSTM (Bidirectional LSTM) architecture
- GRU (Gated Recurrent Units)
- CNN-based sequence models
- Sequence padding and encoding
- Sequence-to-sequence learning
- F1-score evaluation for NER tasks
- Token-level classification

**Datasets:**

- `conllpp_train.txt` - Training data
- `conllpp_test.txt` - Test data
- `conllpp_dev.txt` - Development/validation data

---

### **TP5: Neural Machine Translation with Transformer**

**File:** `tp5/Neural_Machine_Translation_(Transformer).ipynb`

State-of-the-art machine translation using Transformer architecture.

**Key Topics:**

- Transformer architecture overview
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- Encoder-Decoder architecture
- Training a translation model
- Inference and beam search
- BLEU score evaluation

---

## 🗂️ Directory Structure

```
nlp/
├── 📄 README.md                                    # This file
│
├── 📁 tp1/  ............................ Text Preprocessing
│   ├── tp1.ipynb                        # Notebook
│   └── spooky.csv                       # Raw dataset
│
├── 📁 tp2/  ............................ Feature Extraction
│   ├── tp2.ipynb                        # Notebook
│   └── spooky_cleaned.csv               # Cleaned dataset (from TP1)
│
├── 📁 tp3/  ............................ Advanced Engineering
│   ├── tp3.ipynb                        # Notebook
│   └── spooky_cleaned.csv               # Dataset reference
│
├── 📁 tp4/  ............................ Sequence Labeling / NER
│   ├── tp4.ipynb                        # Notebook
│   ├── conllpp_train.txt                # Training set
│   ├── conllpp_test.txt                 # Test set
│   └── conllpp_dev.txt                  # Validation set
│
└── 📁 tp5/  ............................ Machine Translation
    ├── Neural_Machine_Translation_(Transformer).ipynb
    └── en-fr.txt                        # EN-FR parallel corpus
```

---

## 📊 Dataset Reference

<table>
<tr>
<th>Lab</th>
<th>Dataset</th>
<th>Type</th>
<th>Size</th>
<th>Purpose</th>
<th>Source Format</th>
</tr>
<tr>
<td><strong>TP1</strong></td>
<td>spooky.csv</td>
<td>Text Classification</td>
<td>~20K samples</td>
<td>Author identification</td>
<td>CSV (text, author)</td>
</tr>
<tr>
<td><strong>TP2</strong></td>
<td>spooky_cleaned.csv</td>
<td>Processed Text</td>
<td>~20K samples</td>
<td>Feature extraction training</td>
<td>CSV (processed)</td>
</tr>
<tr>
<td><strong>TP3</strong></td>
<td>spooky_cleaned.csv</td>
<td>Processed Text</td>
<td>~20K samples</td>
<td>Cross-validation pipeline</td>
<td>CSV (processed)</td>
</tr>
<tr>
<td><strong>TP4</strong></td>
<td>CoNLL++ (*_*.txt)</td>
<td>Sequence Labeling</td>
<td>~15K sentences</td>
<td>Named Entity Recognition</td>
<td>CoNLL format (word POS NER)</td>
</tr>
<tr>
<td><strong>TP5</strong></td>
<td>en-fr.txt</td>
<td>Parallel Corpus</td>
<td>~100K pairs</td>
<td>Machine translation</td>
<td>TSV (EN \t FR)</td>
</tr>
</table>

---

## 📦 Installation & Setup

### System Requirements

| Component   | Specification                                     |
| :---------- | :------------------------------------------------ |
| **Python**  | 3.7 or higher                                     |
| **Memory**  | Minimum 4GB RAM (8GB+ recommended)                |
| **Storage** | ~2GB for datasets and models                      |
| **GPU**     | Optional (CUDA 11.0+ for TensorFlow acceleration) |

### Dependencies

<table>
<tr>
<td>

**Core Data Science**

- pandas
- numpy
- scikit-learn

</td>
<td>

**Deep Learning**

- TensorFlow 2.x
- Keras

</td>
<td>

**NLP & Utilities**

- nltk
- matplotlib
- jupyter

</td>
</tr>
</table>

### Installation Instructions

**Step 1: Create Virtual Environment (Recommended)**

```bash
python -m venv nlp-env
source nlp-env/bin/activate  # On Windows: nlp-env\Scripts\activate
```

**Step 2: Install Dependencies**

```bash
pip install --upgrade pip
pip install pandas numpy scikit-learn tensorflow nltk matplotlib jupyter matplotlib seaborn
```

**Step 3: Download NLTK Resources**

```python
python -c "import nltk; nltk.download('wordnet')"
```

**Step 4: Verify Installation**

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

---

## 🚀 Quick Start

1. Clone/download the repository
2. Install requirements (see [Installation](#-installation--setup))
3. Navigate to desired lab directory
4. Open notebook with preferred tool

**Starting with Lab 1:**

```bash
cd tp1
jupyter notebook tp1.ipynb
```

**Execution Environments:**

| Environment      | Command                     | Notes                                   |
| :--------------- | :-------------------------- | :-------------------------------------- |
| **Jupyter Lab**  | `jupyter lab tp1/tp1.ipynb` | Recommended for interactive development |
| **VS Code**      | Open `.ipynb` directly      | Requires Jupyter extension              |
| **Google Colab** | Upload notebook to Colab    | Free GPU/TPU access                     |
| **Command Line** | `nbconvert --execute ...`   | Batch processing                        |

---

## 📚 Learning Progression

```
Text Preprocessing (TP1)
    ↓
Feature Extraction (TP2)
    ↓
Advanced Engineering (TP3)
    ↓
Sequence Labeling/NER (TP4)
    ↓
Sequence-to-Sequence (TP5)
```

Each lab builds on concepts from previous labs, progressing from basic text cleaning to advanced deep learning architectures.

```
┌─────────────────────────────────────────────────────────────────┐
│                  NLP LEARNING PROGRESSION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  TP1: Text Preprocessing      [Foundation]                       │
│    ↓                          (Text normalization, cleaning)      │
│  TP2: Feature Extraction      [Classical ML]                      │
│    ↓                          (BoW, TF-IDF, neural networks)      │
│  TP3: Advanced Engineering    [Optimization]                      │
│    ↓                          (Cross-validation, feature select)  │
│  TP4: Sequence Models         [Deep Learning]                     │
│    ↓                          (RNNs, LSTMs, embeddings)           │
│  TP5: Transformers            [State-of-the-Art]                  │
│                               (Attention, seq2seq, translation)   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Recommended Approach:**

- **Sequential Learning:** Follow TP1→TP5 in order for cumulative understanding
- **Thematic Focus:** Skip to relevant labs if targeting specific topics
- **Practice-Oriented:** Run all code cells and experiment with hyperparameters

**Time Commitment:** ~40-50 hours for complete series (self-paced)

---

## 📊 Curriculum Overview

| Lab | Dataset                   | Type                | Purpose                     |
| --- | ------------------------- | ------------------- | --------------------------- |
| TP1 | spooky.csv                | Text Classification | Author identification       |
| TP2 | spooky_cleaned.csv        | Processed Text      | Feature extraction training |
| TP3 | spooky_cleaned.csv        | Processed Text      | Cross-validation pipeline   |
| TP4 | CoNLL++ (conllpp\_\*.txt) | Sequence Labeling   | NER task                    |
| TP5 | en-fr.txt                 | Parallel Corpus     | Machine translation         |

## 💡 Tips for Success

| Best Practice                 | Implementation                                             |
| :---------------------------- | :--------------------------------------------------------- |
| **Understand Before Running** | Read all markdown cells and comments before executing code |
| **Experiment Actively**       | Modify hyperparameters, test different approaches          |
| **Document Learning**         | Keep notes on key concepts and implementations             |
| **Handle Errors**             | Debug systematically; Google error messages                |
| **Verify Results**            | Compare outputs with expected results in documentation     |
| **Memory Management**         | Clear kernel regularly for large computations              |

---

## 🔗 Additional Resources

- **TensorFlow/Keras Documentation:** https://www.tensorflow.org/api_docs
- **Scikit-learn Guide:** https://scikit-learn.org/stable/documentation.html
- **NLTK Book:** https://www.nltk.org/book/
- **Stanford NLP:** https://nlp.stanford.edu/
- **Papers with Code:** https://paperswithcode.com/

---

## ✅ Checklist for Completion

- [ ] TP1: Text preprocessing notebook executed successfully
- [ ] TP2: Neural classification model trained and evaluated
- [ ] TP3: Cross-validation pipeline implemented
- [ ] TP4: BiLSTM NER model trained on CoNLL++ data
- [ ] TP5: Transformer translation model trained (basic)
- [ ] Experimented with hyperparameters in at least 2 labs
- [ ] Documented insights and learnings

---

## 📧 Support & Questions

For technical issues:

1. Review error messages carefully
2. Check notebook comments and markdown sections
3. Verify all dependencies are installed
4. Consult lab-specific documentation
5. Review similar implementations online

---

<div align="center">

**Created:** November 2025  
**Last Updated:** November 15, 2025  
**Status:** ✅ Production-Ready

---

**Happy Learning! 🚀 NLP Enthusiasts**

_This series is designed for continuous learning. Revisit labs as your skills advance._

</div>
