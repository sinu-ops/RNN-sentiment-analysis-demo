# RNN-sentiment-analysis-demo
# 🎬 IMDB Sentiment Analysis Using RNN

A deep learning project that uses a **Recurrent Neural Network (RNN)** to classify movie reviews as **positive or negative**.

The project demonstrates the complete workflow of a basic Natural Language Processing (NLP) classification task — from loading and exploring text data to preprocessing, model training, evaluation, and making predictions on custom reviews.

---

## 📌 Project Overview

Sentiment analysis is a Natural Language Processing task used to determine the emotional tone of text.

In this project, an RNN is trained on the **IMDB Movie Reviews Dataset** to classify movie reviews into two categories:

| Label | Sentiment |
| ----: | --------- |
|   `0` | Negative  |
|   `1` | Positive  |

### 🎯 Objective

Build a basic RNN model capable of learning patterns in sequences of words and predicting whether a movie review expresses a positive or negative sentiment.

---

## 🧠 Why RNN?

Traditional machine learning models generally treat text as a collection of features and may not naturally capture the order of words.

RNNs are designed to process **sequential data**.

For example:

```text
"The movie was not good"
```

The order of the words matters significantly.

An RNN processes the sequence step-by-step and maintains information from previous words, making it suitable for tasks involving sequential data such as:

* Text classification
* Sentiment analysis
* Time-series prediction
* Language modeling
* Sequence generation

This project uses a `SimpleRNN` layer to introduce the fundamental concepts of recurrent neural networks.

---

# 📊 Dataset

The project uses the **IMDB Movie Reviews Dataset** provided through TensorFlow/Keras.

### Dataset Details

* **50,000** total movie reviews
* **25,000** training samples
* **25,000** testing samples
* Binary sentiment classification
* Balanced positive and negative classes

The dataset is loaded directly using:

```python
from tensorflow.keras.datasets import imdb
```

No manual dataset download is required.

---

# 🔄 Project Workflow

```text
IMDB Dataset
     │
     ▼
Load Reviews
     │
     ▼
Explore Dataset
     │
     ▼
Integer Encoding
     │
     ▼
Sequence Padding
     │
     ▼
Embedding Layer
     │
     ▼
Simple RNN
     │
     ▼
Dropout
     │
     ▼
Dense Layer
     │
     ▼
Sigmoid Output
     │
     ▼
Positive / Negative
```

---

# 🏗️ Model Architecture

The neural network consists of the following layers:

```text
Input Sequence
      │
      ▼
Embedding Layer
      │
      ▼
SimpleRNN (64 units)
      │
      ▼
Dropout (30%)
      │
      ▼
Dense Layer (32 units)
      │
      ▼
Sigmoid Output
      │
      ▼
Sentiment Prediction
```

### Model Configuration

| Component               | Configuration       |
| ----------------------- | ------------------- |
| Vocabulary Size         | 10,000              |
| Maximum Sequence Length | 200                 |
| Embedding Dimension     | 128                 |
| RNN Units               | 64                  |
| Dense Units             | 32                  |
| Dropout                 | 0.3                 |
| Output Activation       | Sigmoid             |
| Optimizer               | Adam                |
| Loss Function           | Binary Crossentropy |
| Batch Size              | 64                  |
| Epochs                  | 5                   |

---

# 🔧 Data Preprocessing

The raw reviews are represented as sequences of integers.

For example:

```text
[1, 14, 22, 16, 43, 530, ...]
```

Each integer represents a word in the IMDB vocabulary.

### 1. Vocabulary Limitation

Only the top **10,000 most frequent words** are used.

```python
VOCAB_SIZE = 10000
```

This reduces the complexity of the model and keeps the vocabulary manageable.

### 2. Sequence Padding

Movie reviews have different lengths.

For example:

```text
Review 1 → 120 words
Review 2 → 250 words
Review 3 → 180 words
```

Neural networks require consistent input dimensions.

Therefore, every review is converted to a sequence of length **200**.

```python
MAX_LENGTH = 200
```

Sequences longer than 200 tokens are truncated, while shorter sequences are padded.

---

# 🧪 Exploratory Data Analysis

Before training the model, the project analyzes the dataset.

The notebook explores:

* Dataset size
* Label distribution
* Review lengths
* Minimum review length
* Maximum review length
* Average review length

A histogram is also used to visualize the distribution of review lengths.

Example:

```text
Review Length Distribution

Number of Reviews
       │
       │        ███
       │      ███████
       │    ██████████
       │  █████████████
       └────────────────────
          Review Length
```

---

# 🚀 Model Training

The model is trained using:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)
```

During training, both training and validation performance are monitored.

The project visualizes:

### Training vs Validation Accuracy

```text
Accuracy
   │
   │       ╭──────── Training
   │     ╭─╯
   │   ╭─╯
   │ ╭─╯
   └────────────────────
          Epochs
```

### Training vs Validation Loss

Loss curves are also plotted to identify potential overfitting or underfitting.

---

# 📈 Model Evaluation

After training, the model is evaluated using the test dataset.

The project uses multiple evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Classification Report

```text
              precision    recall    f1-score

Negative        ...         ...        ...
Positive        ...         ...        ...

accuracy                             ...
```

> The exact metrics may vary depending on the training environment, TensorFlow version, and model training run.

---

# 🔲 Confusion Matrix

A confusion matrix provides a detailed view of the model's predictions.

```text
                    Predicted
                 Negative Positive
Actual Negative     TN       FP
       Positive     FN       TP
```

This helps identify:

* Correctly classified negative reviews
* Correctly classified positive reviews
* False positives
* False negatives

---

# 🔮 Custom Prediction

After training, the model can be used to classify new movie reviews.

Example:

```text
Input:
"This movie was absolutely amazing. I loved every minute of it."

Output:
Positive
```

Another example:

```text
Input:
"The movie was boring and disappointing."

Output:
Negative
```

The model produces a probability between `0` and `1`.

```text
Probability >= 0.5 → Positive
Probability < 0.5  → Negative
```

---

# 💻 Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* Recurrent Neural Networks

### NLP

* Text Tokenization
* Integer Encoding
* Sequence Padding
* Word Embeddings
* Sentiment Classification

### Data Science

* NumPy
* Matplotlib
* Scikit-learn

### Development Environment

* Google Colab
* Jupyter Notebook
* Git
* GitHub

---

# 📂 Project Structure

```text
rnn-sentiment-analysis/
│
├── RNN_Sentiment_Analysis.ipynb
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/rnn-sentiment-analysis.git
```

Navigate into the project:

```bash
cd rnn-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The project is designed to run in **Google Colab**.

### Option 1 — Open the Notebook

Open:

```text
RNN_Sentiment_Analysis.ipynb
```

in Google Colab.

### Option 2 — Open from GitHub

Upload the notebook to Google Colab or open it directly using:

```text
https://colab.research.google.com/
```

Then run the notebook cells sequentially.

---

# 📦 Requirements

The main dependencies are:

```text
tensorflow
numpy
matplotlib
scikit-learn
```

You can install them using:

```bash
pip install -r requirements.txt
```

---

# 📚 Key Concepts Learned

This project helped demonstrate the complete workflow of a basic NLP deep learning problem.

### Data Processing

* Loading datasets
* Exploring text data
* Integer encoding
* Sequence padding

### Deep Learning

* Neural network architecture
* Embedding layers
* Recurrent Neural Networks
* Dense layers
* Dropout
* Activation functions

### Model Training

* Train/validation split
* Loss functions
* Optimizers
* Batch processing
* Epochs

### Model Evaluation

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

# ⚠️ Limitations

Although the project successfully demonstrates the fundamentals of RNN-based sentiment analysis, it has some limitations:

* `SimpleRNN` can struggle with long-term dependencies.
* The model uses a relatively small vocabulary.
* Reviews are truncated to 200 tokens.
* The model does not understand language in the same way modern Transformer models do.
* The model is trained only for binary sentiment classification.

---

# 🔮 Future Improvements

The project can be extended in several ways.

### 1. Replace SimpleRNN with LSTM

```text
SimpleRNN
    ↓
LSTM
```

LSTMs are better at learning long-term dependencies.

### 2. Try GRU

Compare:

```text
SimpleRNN vs LSTM vs GRU
```

### 3. Bidirectional RNN

Use:

```text
Bidirectional(LSTM)
```

to process information from both directions.

### 4. Hyperparameter Tuning

Experiment with:

* Number of RNN units
* Embedding dimension
* Batch size
* Learning rate
* Dropout
* Sequence length

### 5. Build a Streamlit Application

Create an interactive interface where users can enter a movie review and receive a sentiment prediction.

### 6. Deploy the Model

The application could eventually be deployed using platforms such as:

* Streamlit Community Cloud
* Hugging Face Spaces
* Other cloud platforms

### 7. Compare with Transformer Models

A future version could compare the RNN with modern NLP architectures such as:

```text
RNN
 ↓
LSTM
 ↓
GRU
 ↓
Bidirectional LSTM
 ↓
Transformer
```

---

# 🎯 Learning Progression

This project is part of a progression toward more advanced NLP and deep learning systems:

```text
Basic Neural Networks
        ↓
Simple RNN
        ↓
LSTM
        ↓
GRU
        ↓
Bidirectional RNN
        ↓
Attention
        ↓
Transformers
        ↓
Large Language Models
```

---

# 👨‍💻 Author

**Sagar Pravin Sapkale**

Engineering Student | Machine Learning & Data Science Enthusiast

Interested in:

* Machine Learning
* Deep Learning
* Natural Language Processing
* Data Science
* AI Engineering

---

# ⭐ Acknowledgements

* TensorFlow / Keras for providing the IMDB dataset and deep learning framework.
* The IMDB dataset for providing the movie review data used in this project.

---

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub!

More machine learning and deep learning projects will be added as I continue learning and building.
