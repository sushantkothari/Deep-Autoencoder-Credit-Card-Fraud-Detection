# Deep Autoencoder Credit Card Fraud Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-darkred?style=for-the-badge" alt="Keras"/>
  <img src="https://img.shields.io/badge/Anomaly%20Detection-Unsupervised-purple?style=for-the-badge" alt="Anomaly Detection"/>
  <img src="https://img.shields.io/badge/Isolation%20Forest-Ensemble-success?style=for-the-badge" alt="Isolation Forest"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<p align="center">
  <strong>An Unsupervised Deep Learning Approach to Financial Anomaly Detection</strong>
</p>

<p align="center">
  Deep Autoencoders · Reconstruction Error Analysis · Gaussian Scoring · Isolation Forest
</p>

---

## Overview

**Deep Autoencoder Credit Card Fraud Detection** is a machine learning project that explores unsupervised anomaly detection for identifying potentially fraudulent credit card transactions.

The primary model is a fully connected Deep Autoencoder trained on legitimate transactions. It learns to reconstruct normal transaction patterns by compressing input features into a lower-dimensional latent representation and then reconstructing the original feature vector.

Transactions with unusually high reconstruction errors can be flagged for further investigation.

The project also explores two complementary approaches:

- **Gaussian Error Modeling:** Models the distribution of reconstruction errors from legitimate transactions.
- **Isolation Forest Ensemble:** Combines reconstruction-based anomaly scores with Isolation Forest scores.

The notebook covers data preprocessing, model training, anomaly scoring, threshold-based classification, evaluation metrics, and visual analysis.

> **Project scope:** This is an experimental machine learning project for learning and research. It is not a validated production fraud prevention system.

## Key Features

- Unsupervised Deep Autoencoder training on legitimate transactions
- Compact neural network with an eight-dimensional latent representation
- Robust scaling of transaction amounts
- Log-Cosh reconstruction loss
- LeakyReLU and ReLU activation functions
- Batch Normalization and Dropout regularization
- Adam optimizer with learning-rate scheduling
- Early stopping and model checkpointing
- Reconstruction error-based anomaly detection
- Gaussian modeling of normal reconstruction errors
- Isolation Forest-based anomaly detection
- Weighted combination of anomaly scores
- Precision, recall, F1-score, and confusion matrix analysis
- ROC and Precision–Recall curve visualization
- Reconstruction error distribution analysis
- PCA-based latent-space visualization
- Reproducible experimentation using fixed random seeds

## Table of Contents

- [Project Motivation](#project-motivation)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Deep Autoencoder Architecture](#deep-autoencoder-architecture)
- [Training Configuration](#training-configuration)
- [Anomaly Detection Methods](#anomaly-detection-methods)
- [Evaluation Framework](#evaluation-framework)
- [Installation](#installation)
- [Usage](#usage)
- [Technology Stack](#technology-stack)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Author](#author)

---

## Project Motivation

Credit card fraud detection is challenging because fraudulent transactions account for only a small fraction of all transactions. Under such severe class imbalance, a model can achieve high overall accuracy while failing to identify fraudulent activity.

Unsupervised anomaly detection offers an alternative to directly learning fraud labels. Instead, the model learns patterns associated with legitimate transactions and identifies transactions that deviate from those patterns.

This project investigates how reconstruction-based anomaly detection can be combined with statistical modeling and a tree-based anomaly detector.

### Potential Applications

- Financial transaction anomaly detection research
- Suspicious transaction screening
- Payment security analytics
- Banking fraud monitoring experiments
- Anomaly detection benchmarking
- Unsupervised learning experimentation

These are potential applications of the techniques explored in this repository, not claims of production deployment or validated real-world performance.

---

## Repository Structure

```text
Deep-Autoencoder-Credit-Card-Fraud-Detection/
│
├── DeepAutoEncoder_Fraud_Detection.ipynb
│   └── Notebook-based implementation and experiments
│
├── best_autoencoder.h5
│   └── Saved Deep Autoencoder checkpoint
│
├── README.md
│   └── Project documentation
│
└── LICENSE
    └── MIT License
```

The dataset must be downloaded separately because it is not included in the repository.

---

## Dataset

This project uses the publicly available Credit Card Fraud Detection dataset hosted on Kaggle.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized numerical transaction features and a binary target indicating whether a transaction is legitimate or fraudulent.

### Dataset Characteristics

| Property | Description |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud prevalence | Approximately 0.173% |
| Original transaction features | 30 |
| Target column | `Class` |

### Feature Description

| Feature | Description |
|---|---|
| `Time` | Time elapsed between the first transaction and the recorded transaction |
| `Amount` | Transaction amount |
| `V1`–`V28` | Anonymized numerical features obtained through PCA |
| `Class` | Binary label identifying legitimate and fraudulent transactions |

### Target Labels

| Label | Meaning |
|---|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

### Data Preparation

The notebook loads the dataset using the following path:

```python
df = pd.read_csv("/content/creditcard.csv")
```

For Google Colab, upload `creditcard.csv` to the expected location. For local execution, update the path to match the directory containing your dataset.

The preprocessing workflow includes:

1. Loading and inspecting the dataset.
2. Checking for missing values.
3. Removing the `Time` column.
4. Scaling `Amount` using `RobustScaler`.
5. Separating input features from the target label.
6. Splitting the data into training and testing subsets.
7. Selecting legitimate transactions for Autoencoder training.

After removing `Time`, the model uses 29 input features.

**Important:** The `Class` column is used for evaluation and should not be included in the Autoencoder's input features.

---

## System Architecture

The overall workflow follows these stages:

1. Load the credit card transaction dataset.
2. Inspect data quality and class distribution.
3. Preprocess the numerical features.
4. Train the Deep Autoencoder on legitimate transactions.
5. Reconstruct the input transaction features.
6. Calculate reconstruction errors.
7. Investigate reconstruction error thresholding.
8. Explore Gaussian-based anomaly scoring.
9. Train and evaluate an Isolation Forest detector.
10. Combine anomaly scores using a weighted ensemble.
11. Evaluate predictions using classification metrics.
12. Generate diagnostic visualizations.
13. Explore latent representations using PCA.
14. Save the trained Autoencoder checkpoint.

### Detection Pipeline

```text
Credit Card Transactions
          |
          v
   Data Preprocessing
          |
          v
    Feature Scaling
          |
          v
    Deep Autoencoder
          |
          v
  Reconstructed Features
          |
          v
   Reconstruction Error
          |
          +--------------------------+
          |                          |
          v                          v
  Error Thresholding        Gaussian Error Modeling
          |                          |
          +-------------+------------+
                        |
                        v
                Isolation Forest
                        |
                        v
             Weighted Score Ensemble
                        |
                        v
                Anomaly Classification
                        |
                        v
             Potential Fraud Detection
```

The Gaussian scoring method and the ensemble represent separate experimental detection strategies. Their scores should not be interpreted as calibrated probabilities that a transaction is fraudulent.

---

## Deep Autoencoder Architecture

The Deep Autoencoder consists of three main components:

- **Encoder:** Compresses the input features into a lower-dimensional representation.
- **Latent representation:** Captures a compact representation of the input.
- **Decoder:** Reconstructs the original feature vector from the latent representation.

### Encoder

```text
Input: 29 Features
        |
        v
Dense Layer: 64 Units
        |
        v
Dense Layer: 32 Units
        |
        v
Dense Layer: 16 Units
        |
        v
Latent Representation: 8 Units
```

### Decoder

```text
Latent Representation: 8 Units
        |
        v
Dense Layer: 16 Units
        |
        v
Dense Layer: 32 Units
        |
        v
Dense Layer: 64 Units
        |
        v
Output: 29 Reconstructed Features
```

### Architectural Configuration

| Component | Configuration |
|---|---|
| Input dimension | 29 |
| Encoder dimensions | 64 → 32 → 16 → 8 |
| Latent dimension | 8 |
| Decoder dimensions | 16 → 32 → 64 → 29 |
| Encoder activation | LeakyReLU |
| Decoder activation | ReLU |
| Output activation | Linear |
| Normalization | Batch Normalization |
| Regularization | Dropout |
| Dropout rate | 0.2 |
| Loss function | Log-Cosh |
| Optimizer | Adam |

The compressed latent representation encourages the model to learn a compact representation of legitimate transaction patterns.

### Reconstruction Error

After training, the Autoencoder reconstructs transaction feature vectors. The difference between each original feature vector and its reconstruction is used to calculate a reconstruction error.

The notebook uses mean squared error across the input features for each transaction.

In plain terms:

```text
Reconstruction Error =
    Mean of the squared differences between
    the original and reconstructed feature values
```

Where:

- **Original features:** The transaction values supplied to the model.
- **Reconstructed features:** The values produced by the decoder.
- **Reconstruction error:** A numerical measure of how different the reconstruction is from the original input.

A larger reconstruction error indicates that a transaction is more difficult for the trained model to reconstruct. It can therefore serve as an anomaly score, but it is not a direct probability of fraud.

---

## Training Configuration

The notebook uses the following training configuration.

| Parameter | Value |
|---|---|
| Random seed | `42` |
| Input features | 29 |
| Latent dimension | 8 |
| Batch size | 256 |
| Maximum epochs | 200 |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Loss function | Log-Cosh |
| Dropout rate | 0.2 |
| Training population | Legitimate transactions |
| Model checkpoint | `best_autoencoder.h5` |

### Training Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| EarlyStopping | `patience=10`, `restore_best_weights=True` | Stops training when validation loss stops improving |
| ReduceLROnPlateau | `factor=0.5`, `patience=5`, `min_lr=1e-6` | Reduces the learning rate when validation loss plateaus |
| ModelCheckpoint | `save_best_only=True` | Saves the best checkpoint according to the monitored validation loss |

The model is compiled with the Log-Cosh loss function and the Adam optimizer.

---

## Anomaly Detection Methods

### 1. Reconstruction Error Thresholding

The primary detection strategy uses reconstruction error as an anomaly score.

Transactions with reconstruction errors above a selected threshold are classified as potential anomalies.

The general procedure is:

1. Reconstruct transaction features using the trained Autoencoder.
2. Calculate a reconstruction error for each transaction.
3. Examine the distribution of reconstruction errors.
4. Select a decision threshold.
5. Flag transactions whose errors exceed that threshold.
6. Evaluate the resulting predictions using fraud-oriented metrics.

Threshold selection involves a trade-off:

- A lower threshold can identify more anomalies but may flag more legitimate transactions.
- A higher threshold can reduce false alarms but may miss fraudulent transactions.

**Evaluation consideration:** Threshold selection should be performed using a dedicated validation set. The final test set should remain untouched until the detection strategy and threshold have been finalized.

### 2. Gaussian Reconstruction Error Modeling

The notebook fits a Gaussian distribution to reconstruction errors from legitimate transactions.

The model estimates two parameters:

- **Mean (`mu`):** The average reconstruction error for the modeled legitimate transactions.
- **Standard deviation (`sigma`):** The spread of those reconstruction errors.

The implemented score is calculated as:

```python
score = 1 - norm.cdf((reconstruction_error - mu) / sigma)
```

Here, `norm.cdf` is the cumulative distribution function of the standard normal distribution.

The notebook applies a cutoff of `0.995` to this score.

**Important implementation limitation:** The expression above calculates an upper-tail probability in reverse. As reconstruction error increases, the score decreases. This conflicts with the intended interpretation of a higher-is-more-anomalous score.

Consequently, the current probabilistic scoring implementation requires correction and reevaluation before its effectiveness can be established.

These scores should not be interpreted as calibrated probabilities of fraud.

### 3. Isolation Forest

Isolation Forest is a tree-based anomaly detection algorithm that identifies observations that can be isolated relatively easily from the rest of the dataset.

The notebook configures the detector as follows:

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `contamination` | 0.0017 |
| `random_state` | 42 |

The implementation fits Isolation Forest on the training features and calculates anomaly scores for the test features.

Isolation Forest provides a complementary anomaly detection signal that can be investigated alongside Autoencoder reconstruction errors.

### 4. Autoencoder–Isolation Forest Ensemble

The ensemble combines two anomaly signals:

- **Autoencoder:** Uses reconstruction error to measure how difficult a transaction is to reconstruct.
- **Isolation Forest:** Uses tree-based isolation to identify potentially anomalous observations.

The notebook normalizes the reconstruction errors and Isolation Forest scores using min-max scaling, then combines them with the following weighted score:

```python
final_score = 0.6 * rec_norm + 0.4 * iso_norm
```

The weights assign 60% of the combined score to the normalized reconstruction error and 40% to the normalized Isolation Forest score.

The notebook selects a decision threshold using the 99.5th percentile of ensemble scores for legitimate transactions in the test set.

**Evaluation limitation:** Selecting this threshold using test-set labels introduces evaluation leakage. Score normalization using test-set score ranges can also make the evaluation dependent on the test data.

For a more reliable assessment, fit normalization parameters and select the threshold using training or validation data, then evaluate the finalized method on a separate, untouched test set.

---

## Evaluation Framework

The notebook explores classification metrics and diagnostic visualizations to assess the different anomaly detection approaches.

### Metrics

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of flagged transactions that are actually fraudulent |
| Recall | Measures the proportion of fraudulent transactions successfully identified |
| F1-score | Combines precision and recall into a single metric |
| ROC-AUC | Evaluates ranking-based discrimination across decision thresholds |
| Precision–Recall Curve | Examines the precision–recall trade-off, particularly under class imbalance |
| Confusion Matrix | Summarizes correct and incorrect classifications |

Because fraudulent transactions are rare in this dataset, precision and recall are particularly important when assessing detection behavior. Overall accuracy alone is not sufficient.

### Visual Analysis

The notebook includes visual analysis of:

- Transaction class distribution
- Feature distributions
- Training and validation loss
- Reconstruction error distributions
- Fraud detection thresholds
- Confusion matrices
- ROC curves
- Precision–Recall curves
- Latent-space projections using PCA

These visualizations support model diagnostics and help identify the limitations of individual detection strategies.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sushantkothari/Deep-Autoencoder-Credit-Card-Fraud-Detection.git

cd Deep-Autoencoder-Credit-Card-Fraud-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn jupyter
```

Use Python and TensorFlow versions that are compatible with each other and with the environment in which you intend to execute the notebook.

---

## Usage

### 1. Obtain the Dataset

Download the dataset from:

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` in the expected working directory, or update the dataset path in the notebook.

The original notebook uses:

```python
df = pd.read_csv("/content/creditcard.csv")
```

This path is intended for a Google Colab-style environment. Local execution may require a different path.

### 2. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 3. Execute the Notebook

1. Open `DeepAutoEncoder_Fraud_Detection.ipynb`.
2. Confirm that the dataset path is correct.
3. Run the notebook cells sequentially.
4. Review the data preprocessing and class distribution.
5. Train the Deep Autoencoder or load the available checkpoint where compatible.
6. Calculate reconstruction errors.
7. Examine threshold-based anomaly detection.
8. Inspect the Gaussian scoring and Isolation Forest experiments.
9. Review classification metrics and visualizations.

The repository includes `best_autoencoder.h5` as a saved model artifact.

**Note:** A saved model alone may not reproduce the entire pipeline. Reproducible inference can also require the fitted preprocessing objects, expected feature ordering, score normalization parameters, and decision thresholds.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network construction and training |
| NumPy | Numerical computation |
| Pandas | Data loading and manipulation |
| Scikit-learn | Preprocessing, Isolation Forest, PCA, splitting, and evaluation |
| SciPy | Gaussian probability modeling |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Interactive experimentation |

---

## Limitations

The current implementation is experimental and has several limitations to consider:

- The Gaussian scoring formula has a score-direction issue.
- Threshold selection using evaluation-set labels can introduce leakage.
- Test-set score normalization can affect the reliability of ensemble evaluation.
- Anomaly scores are not calibrated probabilities of fraud.
- Class imbalance makes false-positive analysis important.
- A saved model checkpoint does not necessarily include the complete preprocessing and inference pipeline.
- Results from a single split may not generalize to other transaction distributions.

These limitations should be addressed before considering the methods for real-world financial decision-making.

---

## Future Improvements

Potential directions for further development include:

- Correcting the Gaussian anomaly score orientation.
- Separating training, validation, and final test data.
- Calibrating decision thresholds using validation data only.
- Fitting score normalization parameters without using test data.
- Comparing detection methods under consistent evaluation conditions.
- Evaluating precision–recall behavior across thresholds.
- Conducting systematic false-positive analysis.
- Persisting preprocessing objects and inference configuration.
- Adding automated tests for preprocessing and model inference.
- Investigating model export using compatible deployment formats.
- Developing an API for inference experiments.
- Exploring streaming transaction anomaly detection.
- Investigating concept drift and changes in transaction distributions.
- Adding monitoring and explainability tools for anomaly investigation.

---

## Reproducibility

The notebook uses a fixed random seed and includes model checkpointing to support repeatable experimentation.

For a more reproducible end-to-end pipeline, the following artifacts should also be saved:

- Fitted `RobustScaler`
- Expected input feature order
- Trained model and compatible dependency versions
- Selected anomaly thresholds
- Score normalization parameters
- Gaussian scoring configuration

These artifacts are important for ensuring that future inference uses the same preprocessing and scoring behavior as the training pipeline.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**

AI/ML Engineer | Data Scientist | Deep Learning

- **GitHub:** [sushantkothari](https://github.com/sushantkothari)
- **Repository:** [Deep Autoencoder Credit Card Fraud Detection](https://github.com/sushantkothari/Deep-Autoencoder-Credit-Card-Fraud-Detection)
