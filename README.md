 <p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-darkred?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Anomaly%20Detection-Unsupervised-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Fraud%20Detection-FinTech-critical?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Isolation%20Forest-Ensemble-success?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">Deep Autoencoder Credit Card Fraud Detection</h1>

<p align="center">
  <strong>An Unsupervised Deep Learning Framework for Financial Anomaly Detection</strong>
</p>

<p align="center">
  Deep Autoencoders • Reconstruction Error Analysis • Probabilistic Scoring • Isolation Forest Ensemble
</p>

---

## Overview

**Deep Autoencoder Credit Card Fraud Detection** is an unsupervised deep learning project designed to identify potentially fraudulent credit card transactions by learning the characteristics of legitimate financial activity.

The framework uses a lightweight, fully connected Deep Autoencoder trained on normal transactions. By compressing transaction features into a low-dimensional latent representation and reconstructing the original inputs, the model learns patterns associated with legitimate transactions.

Transactions that produce unusually high reconstruction errors can subsequently be flagged as potential anomalies.

To investigate complementary detection strategies, the project extends reconstruction-based detection with Gaussian probability modeling and an Autoencoder–Isolation Forest ensemble.

The implementation covers an end-to-end anomaly detection workflow, including:

- Transaction data inspection and preprocessing
- Robust scaling of monetary features
- Unsupervised Deep Autoencoder training
- Latent representation learning
- Reconstruction error computation
- Statistical anomaly scoring
- Isolation Forest-based detection
- Weighted anomaly score aggregation
- Fraud-oriented classification evaluation
- ROC and Precision–Recall analysis
- Reconstruction error visualization
- Latent-space exploration using PCA
- Model checkpointing and reproducible experimentation

The project demonstrates how deep representation learning and classical anomaly detection techniques can be combined to investigate fraud detection under extreme class imbalance.

---

## Why This Project Matters

Credit card fraud detection is challenging because fraudulent transactions represent only a small fraction of the overall transaction volume. A model can achieve deceptively high accuracy while failing to identify most fraudulent transactions.

Unsupervised anomaly detection approaches this problem differently: instead of learning fraud patterns directly, the model learns the structure of legitimate transactions and identifies observations that deviate from those patterns.

This project explores how a Deep Autoencoder can support that process and how complementary anomaly scoring methods can be evaluated alongside it.

### Potential Real-World Applications

- Credit card transaction anomaly detection
- Banking fraud monitoring research
- Payment processing risk analysis
- Suspicious transaction screening
- Financial security analytics
- Unsupervised learning experimentation
- Transaction risk assessment pipelines
- Fraud detection model benchmarking

*This repository is an experimental machine learning project, not a validated production fraud prevention system.*

---

## Key Technical Highlights

- Unsupervised Deep Autoencoder trained on legitimate transactions
- Compact neural architecture with approximately 9.8K parameters
- Eight-dimensional latent representation
- Robust monetary feature scaling with `RobustScaler`
- Log-Cosh reconstruction loss
- Batch Normalization and Dropout regularization
- LeakyReLU encoder activations
- ReLU decoder activations
- Adam optimization with adaptive learning rate scheduling
- Early stopping and model checkpointing
- Reconstruction error-based anomaly detection
- Gaussian-based probabilistic scoring
- Isolation Forest anomaly detection
- Weighted Autoencoder–Isolation Forest ensemble
- Fraud-oriented precision, recall, and F1-score evaluation
- ROC and Precision–Recall curves
- Confusion matrix and reconstruction error diagnostics
- PCA-based latent-space visualization
- Fixed random seed for reproducibility

---

## Research Contributions

The project explores the following machine learning approaches:

- **Normal-behavior representation learning:** Training an Autoencoder on legitimate transactions to model normal transaction patterns.
- **Reconstruction-based anomaly detection:** Using deviations between original and reconstructed features as an anomaly indicator.
- **Probabilistic error modeling:** Fitting a Gaussian distribution to legitimate transaction reconstruction errors.
- **Hybrid anomaly scoring:** Combining reconstruction errors with Isolation Forest scores.
- **Threshold-based detection:** Investigating how anomaly thresholds influence fraud precision and recall.
- **Imbalance-aware evaluation:** Comparing detection strategies using fraud-specific classification metrics.
- **Latent-space analysis:** Projecting learned representations into two dimensions using PCA.

These experiments provide a foundation for comparing unsupervised fraud detection strategies and identifying areas for further improvement.

---

## Core Architectures

The framework explores three primary detection strategies.

| Architecture | Purpose |
|---|---|
| Deep Autoencoder | Learns a compressed representation of legitimate transaction patterns |
| Gaussian Error Model | Estimates the distribution of normal reconstruction errors |
| Isolation Forest | Identifies observations that are easier to isolate from the surrounding data |
| Weighted Ensemble | Combines Autoencoder reconstruction errors and Isolation Forest anomaly scores |

---

## Repository Structure

```text
Deep-Autoencoder-Credit-Card-Fraud-Detection/
│
├── DeepAutoEncoder_Fraud_Detection.ipynb
│   └── Complete notebook-based implementation
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

The transaction dataset is not included in the repository and must be obtained separately.

---

## Dataset

The project uses the publicly available Credit Card Fraud Detection dataset hosted on Kaggle.

**Dataset:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains anonymized transaction features and a binary label indicating whether a transaction is legitimate or fraudulent.

### Dataset Characteristics

| Property | Description |
|---|---|
| Total transactions | 284,807 |
| Legitimate transactions | 284,315 |
| Fraudulent transactions | 492 |
| Fraud prevalence | Approximately 0.173% |
| Original features | 30 transaction features |
| Target column | `Class` |

### Feature Description

| Feature | Description |
|---|---|
| `Time` | Time elapsed between the first transaction and the transaction being recorded |
| `Amount` | Monetary amount of the transaction |
| `V1`–`V28` | Anonymized numerical features obtained through PCA |
| `Class` | Binary transaction label |

### Target Variable

| Label | Meaning |
|---|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

### Dataset Preparation

The notebook reads the dataset from:

```python
df = pd.read_csv("/content/creditcard.csv")
```

To execute the notebook in Google Colab, upload `creditcard.csv` to the expected location or modify the loading path to match your environment.

The preprocessing workflow:

1. Inspects the dataset and checks for missing values.
2. Removes the `Time` column.
3. Scales `Amount` using `RobustScaler`.
4. Retains the anonymized PCA features.
5. Splits the data into training and testing subsets.
6. Separates legitimate transactions for Autoencoder training.

After dropping `Time`, the model uses 29 input features.

**Important:** The `Class` column is retained for evaluation, not as an Autoencoder input. Training on legitimate transactions only is essential to the intended unsupervised learning strategy.

---

## System Architecture

### End-to-End Pipeline

1. Load the credit card transaction dataset.
2. Inspect data quality and class distribution.
3. Remove the `Time` feature.
4. Scale transaction amounts using `RobustScaler`.
5. Split the dataset into training and testing subsets.
6. Train the Autoencoder on legitimate transactions.
7. Reconstruct the transaction feature vectors.
8. Calculate per-transaction reconstruction errors.
9. Apply reconstruction error thresholding.
10. Explore Gaussian-based anomaly scoring.
11. Train an Isolation Forest detector.
12. Normalize and combine anomaly scores.
13. Evaluate predictions using fraud-focused metrics.
14. Generate classification and distribution visualizations.
15. Examine the latent representation using PCA.
16. Save the trained Autoencoder checkpoint.

### Detection Workflow

```text
Credit Card Transactions
          |
          v
  Data Preprocessing
          |
          v
  Robust Feature Scaling
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
          +-----------------------+
          |                       |
          v                       v
   Error Thresholding      Gaussian Error Model
          |                       |
          +-----------+-----------+
                      |
                      v
              Isolation Forest
                      |
                      v
           Weighted Score Ensemble
                      |
                      v
              Anomaly Decision
                      |
                      v
          Potential Fraud Detection
```

The Gaussian scoring branch and ensemble are evaluated as separate experimental strategies in the notebook. Their outputs should not be interpreted as calibrated probabilities of fraud.

---

## Deep Learning Methodology

### 1. Data Preprocessing

The preprocessing pipeline prepares the numerical transaction features for Autoencoder training.

The principal operations include:

- Dataset loading and validation
- Missing-value inspection
- Removal of the `Time` column
- Robust scaling of `Amount`
- Separation of input features and target labels
- Train/test splitting
- Selection of legitimate transactions for training

`RobustScaler` uses robust distribution statistics to reduce the influence of extreme values during scaling.

The original PCA-transformed features are retained, while `Amount` is scaled separately.

### 2. Deep Autoencoder Architecture

The Deep Autoencoder consists of an encoder, a compressed latent representation, and a decoder.

The encoder progressively reduces the dimensionality of the input, while the decoder attempts to reconstruct the original feature vector.

#### Encoder

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
Latent Space: 8 Units
```

#### Decoder

```text
Latent Space: 8 Units
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

#### Architectural Components

| Component | Configuration |
|---|---|
| Input dimension | 29 |
| Encoder | 64 → 32 → 16 → 8 |
| Latent dimension | 8 |
| Decoder | 8 → 16 → 32 → 64 → 29 |
| Encoder activation | LeakyReLU |
| Decoder activation | ReLU |
| Output activation | Linear |
| Normalization | Batch Normalization |
| Regularization | Dropout |
| Dropout rate | 0.2 |
| Loss function | Log-Cosh |
| Optimizer | Adam |

The compressed latent representation encourages the model to capture useful characteristics of legitimate transactions.

### 3. Reconstruction Error

After training, the Autoencoder reconstructs the test transactions.

The notebook calculates the per-transaction mean squared reconstruction error:

\[
E(x)=\frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
\]

Where:

- \(x_i\) is an original feature value.
- \(\hat{x}_i\) is its reconstructed value.
- \(n\) is the number of input features.

Higher reconstruction errors indicate that a transaction is more difficult for the trained model to reconstruct.

This error acts as an anomaly score rather than a direct probability of fraud.

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
| ModelCheckpoint | `best_autoencoder.h5`, `save_best_only=True` | Saves the best-performing checkpoint according to the monitored validation loss |

The model is compiled using the Log-Cosh loss function and the Adam optimizer.

---

## Anomaly Detection Strategies

### 1. Reconstruction Error Thresholding

The primary detection strategy classifies transactions as potential anomalies when their reconstruction errors exceed a selected threshold.

The notebook investigates threshold selection using reconstruction errors from legitimate transactions and evaluates fraud-oriented F1-score across candidate thresholds.

The process involves:

1. Calculating reconstruction errors for test transactions.
2. Examining the distribution of normal transaction errors.
3. Evaluating candidate thresholds.
4. Comparing precision, recall, and F1-score.
5. Applying the selected threshold to generate predictions.

A threshold that is too low may flag many legitimate transactions. A threshold that is too high may allow fraudulent transactions to go undetected.

**Evaluation consideration:** The notebook's threshold search uses labeled evaluation data to select a threshold. For an unbiased final performance estimate, threshold selection should instead be performed on a separate validation set before evaluating once on the held-out test set.

### 2. Probabilistic Autoencoder

The notebook fits a Gaussian distribution to the reconstruction errors of legitimate transactions.

It estimates the mean and standard deviation of normal reconstruction errors:

\[
E \sim \mathcal{N}(\mu,\sigma^2)
\]

The implemented score is:

\[
S(x)=1-\Phi\left(\frac{E(x)-\mu}{\sigma}\right)
\]

Where:

- \(E(x)\) represents the transaction reconstruction error.
- \(\mu\) represents the mean normal reconstruction error.
- \(\sigma\) represents its standard deviation.
- \(\Phi\) represents the standard normal cumulative distribution function.

The notebook applies a cutoff of `0.995` to this score.

**Important limitation:** This formula represents the Gaussian upper-tail probability. As reconstruction error increases, the score decreases. Consequently, the current implementation's score direction is inconsistent with the intended interpretation of a higher-is-more-anomalous score.

The notebook's reported probabilistic results reflect this limitation. The method requires correction and reevaluation before it can be considered an effective detection strategy.

### 3. Autoencoder + Isolation Forest Ensemble

The ensemble combines two complementary anomaly detection signals:

- **Autoencoder:** Measures reconstruction error.
- **Isolation Forest:** Estimates anomaly scores using tree-based isolation.

The notebook configures Isolation Forest with:

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `contamination` | 0.0017 |
| `random_state` | 42 |

The implementation fits Isolation Forest on `X_train` and calculates anomaly scores on `X_test`.

#### Weighted Ensemble

The notebook normalizes the reconstruction errors and Isolation Forest scores using min-max scaling, then combines them:

```python
final_score = 0.6 * rec_norm + 0.4 * iso_norm
```

The decision threshold is selected using the 99.5th percentile of ensemble scores for legitimate test transactions.

```python
final_threshold = np.percentile(
    final_score[error_df.true_class == 0],
    99.5
)

y_pred_ensemble = (
    final_score > final_threshold
).astype(int)
```

The ensemble provides a way to investigate whether combining the two anomaly detection signals improves fraud detection.

However, the current implementation uses test-set labels to identify legitimate samples for threshold selection. This introduces evaluation leakage and means the reported ensemble results should be considered exploratory rather than an unbiased estimate of generalization performance.

---

## Evaluation Framework

The notebook evaluates the three detection strategies using classification reports and visual diagnostics.

### Metrics Used

| Metric | Purpose |
|---|---|
| Precision | Measures the proportion of flagged transactions that are fraudulent |
| Recall | Measures the proportion of actual fraudulent transactions detected |
| F1-score | Balances fraud precision and recall |
| ROC-AUC | Measures ranking-based discrimination across thresholds |
| Precision–Recall Curve | Examines precision and recall under severe class imbalance |
| Confusion Matrix | Summarizes correct and incorrect classifications |

### Visual Analysis

The notebook includes visual analysis of:

- Transaction class distribution
- Feature distributions
- Training and validation loss
- Reconstruction error distributions
- Fraud detection thresholds
- Confusion matrix
- ROC curve
- Precision–Recall curve
- Latent-space projections using PCA

These visualizations support model diagnostics and help identify the limitations of individual detection strategies.

---

## Results Snapshot

The notebook's saved evaluation outputs report the following fraud-class metrics on a test set containing 56,864 legitimate transactions and 98 fraudulent transactions.

### Fraud Detection Performance

| Detection Strategy | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Deep Autoencoder | 0.34 | 0.60 | 0.44 |
| Probabilistic Autoencoder | 0.00 | 0.00 | 0.00 |
| Autoencoder + Isolation Forest | 0.16 | 0.55 | 0.25 |

*Source: Classification reports stored in the notebook's executed outputs.*

### Interpretation

**Deep Autoencoder**

- Achieves a reported fraud recall of 60%.
- Detects a substantial portion of fraudulent transactions at the selected threshold.
- Has a fraud precision of 34%, indicating that many flagged transactions are legitimate.

**Probabilistic Autoencoder**

- Reports zero precision, recall, and F1-score for the fraud class.
- Its current upper-tail score orientation and thresholding logic require correction.
- Should not be interpreted as an effective fraud detector in its current form.

**Autoencoder + Isolation Forest**

- Reports a fraud recall of 55%.
- Achieves a fraud precision of 16%.
- Flags more legitimate transactions relative to the number of correctly identified fraudulent transactions.

### Results Limitations

These metrics are the notebook's reported experimental outputs, not independently reproduced or production-validated results.

In particular:

- Threshold selection uses information from the evaluation set.
- Ensemble score normalization uses test-set score ranges.
- The Isolation Forest contamination parameter is configured using a fixed value.
- The probabilistic score has a directionality issue.
- High overall accuracy is not sufficient to establish effective fraud detection.

The results should therefore be interpreted as a baseline for further experimentation. A separate validation set, corrected scoring logic, and an untouched test set are recommended for a more reliable comparison.

---

## Reproducibility

The notebook establishes reproducibility controls through:

- Fixed random seed initialization
- NumPy random seed configuration
- TensorFlow random seed configuration
- Fixed train/test splitting
- Consistent preprocessing operations
- Model checkpointing
- Repeatable evaluation procedures

For a reproducible deployment pipeline, additional artifacts should also be persisted:

- Fitted `RobustScaler`
- Expected feature ordering
- Selected anomaly thresholds
- Anomaly score normalization parameters
- Corrected Gaussian scoring configuration
- Model and dependency versions

These additional artifacts are not represented as separate files in the current repository structure.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/Deep-Autoencoder-Credit-Card-Fraud-Detection.git

cd Deep-Autoencoder-Credit-Card-Fraud-Detection
```

### Install Dependencies

Install the libraries used by the notebook:

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn jupyter
```

For reproducible execution, use a Python environment compatible with the installed TensorFlow version.

---

## Usage

### Launch Jupyter Notebook

```bash
jupyter notebook
```

### Execute the Notebook

1. Open `DeepAutoEncoder_Fraud_Detection.ipynb`.
2. Download `creditcard.csv` from the linked dataset source.
3. Place the dataset in the expected working directory or update the file path.
4. Run the notebook cells sequentially.
5. Inspect the transaction distribution and preprocessing outputs.
6. Train the Deep Autoencoder or load the available checkpoint where compatible.
7. Calculate reconstruction errors.
8. Evaluate threshold-based anomaly detection.
9. Examine Gaussian scoring and Isolation Forest ensemble outputs.
10. Review classification reports and visualizations.

The repository contains `best_autoencoder.h5` as a saved model artifact.

**Note:** The current notebook is written with a Google Colab-style dataset path. Local Jupyter or other environments may require path adjustments.

---

## Technology Stack

- **Python** — Core programming language
- **TensorFlow** — Deep learning framework
- **Keras** — Neural network construction and training
- **NumPy** — Numerical computation
- **Pandas** — Data loading and preprocessing
- **Scikit-learn** — Robust scaling, Isolation Forest, PCA, data splitting, and evaluation metrics
- **SciPy** — Gaussian probability modeling
- **Matplotlib** — Data visualization and evaluation plots
- **Seaborn** — Statistical visualization and confusion matrix heatmaps
- **Jupyter Notebook** — Interactive development and experimentation

---

## Engineering Principles

- Unsupervised learning on legitimate transactions
- Compact neural network architecture
- Reconstruction-based anomaly scoring
- Complementary anomaly detection methods
- Class-imbalance-aware evaluation
- Reproducible training configuration
- Validation-aware model checkpointing
- Visualization-driven model diagnostics
- Explicit separation of training and evaluation labels
- Transparent reporting of detection limitations

---

## Future Improvements

- Correct the Gaussian anomaly score orientation and reevaluate thresholds.
- Introduce a dedicated validation set for threshold calibration.
- Reserve an untouched test set for final evaluation.
- Fit score normalization parameters on training or validation data.
- Compare Autoencoder and Isolation Forest performance under identical evaluation conditions.
- Evaluate precision–recall performance using continuous anomaly scores.
- Add systematic threshold calibration and false-positive analysis.
- Persist the fitted scaler and calibrated thresholds.
- Introduce automated tests for preprocessing and inference.
- Export compatible model artifacts to ONNX or TensorFlow Lite.
- Develop a FastAPI inference service.
- Explore streaming transaction anomaly detection.
- Investigate concept drift and changing transaction distributions.
- Add monitoring and explainability tools for anomaly investigation.

---

## Environment Requirements

Recommended starting environment:

- Python 3.10+ 
- A compatible TensorFlow/Keras installation
- NumPy and Pandas
- Scikit-learn
- SciPy
- Matplotlib and Seaborn
- Jupyter Notebook

GPU acceleration is optional for this relatively lightweight network. Actual training speed depends on the hardware, dataset, and software environment.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**

AI/ML Engineer | Data Scientist | Deep Learning Researcher

GitHub: [sushantkothari](https://github.com/sushantkothari)

Repository: [Deep-Autoencoder-Credit-Card-Fraud-Detection](https://github.com/sushantkothari/Deep-Autoencoder-Credit-Card-Fraud-Detection)
