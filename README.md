# Deep Autoencoder Credit Card Fraud Detection

## Tagline
Unsupervised anomaly detection for credit card fraud using a lightweight Deep Autoencoder, enhanced with probabilistic scoring and an Isolation Forest ensemble — engineered for real-world class imbalance and production readiness.

---

## Description
This repository contains an end-to-end implementation of an **unsupervised credit card fraud detection system** using a **Deep Autoencoder neural network**.

The model is trained **exclusively on legitimate transactions** and detects potential fraud by analyzing **reconstruction error**, making it well-suited for **highly imbalanced datasets**. To improve robustness across diverse fraud patterns, the project also explores:

- **Probabilistic anomaly scoring** using a Gaussian model over reconstruction errors  
- An **Autoencoder + Isolation Forest ensemble** approach  

The implementation follows production-minded practices, including reproducibility controls, lightweight architecture, and saved model artifacts for deployment.

---

## Dataset
The model is developed using a **publicly available credit card transactions dataset** containing anonymized features derived via PCA.

- The dataset is **not included** in this repository due to licensing constraints.
- Features are anonymized; only `Time`, `Amount`, and `Class` are explicitly defined.

---

## Approach
1. Preprocess transaction data and scale monetary values
2. Train a Deep Autoencoder on **non-fraudulent transactions only**
3. Compute reconstruction error on unseen data
4. Detect fraud using:
   - Reconstruction error thresholding
   - Probabilistic anomaly scoring
   - Autoencoder + Isolation Forest ensemble
5. Evaluate performance using classification metrics and visual analysis

---

## Key Features & Highlights
- Fully **unsupervised learning** to handle extreme class imbalance
- Lightweight autoencoder (~9.8K parameters) for fast training and inference
- Robust preprocessing using `RobustScaler` to handle outliers
- Training safeguards:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint
- Multiple anomaly detection strategies
- Rich evaluation and interpretability tools
- Deployment-friendly design (SavedModel / ONNX / TFLite ready)

---

## Model Architecture
- **Type**: Fully connected Deep Autoencoder
- **Encoder**:  
  `64 → 32 → 16 → 8 (latent space)`
- **Decoder**:  
  `8 → 16 → 32 → 64 → Input`
- **Layers & Techniques**:
  - Batch Normalization
  - LeakyReLU activations (encoder)
  - ReLU activations (decoder)
  - Dropout (0.2) for regularization
- **Loss Function**: Log-Cosh
- **Optimizer**: Adam (lr = 1e-3, adaptive scheduling)

---

## Training Configuration
- **Input features**: 29 (after dropping `Time` and scaling `Amount`)
- **Batch size**: 256
- **Epochs**: Up to 200 (with early stopping)
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)`
  - `ModelCheckpoint('best_autoencoder.h5', save_best_only=True)`
- **Reproducibility**: Global random seed fixed (`RANDOM_SEED = 42`)

---

## Preprocessing
- Drop `Time` column
- Scale `Amount` using `RobustScaler`
- Train/test split with fixed `random_state`
- Autoencoder is trained **only on normal transactions (Class = 0)**
- Labels are used **strictly for evaluation**, never during training

---

## Detection Strategies
### 1. Reconstruction Error Thresholding
- Compute per-sample MSE between input and reconstructed output
- Select threshold from high percentiles (e.g., 99.0–99.9) of normal-class errors
- Threshold tuned to maximize fraud-oriented F1-score

### 2. Probabilistic Autoencoder
- Fit Gaussian distribution on reconstruction errors of normal samples
- Compute anomaly probability:  
  `1 - norm.cdf(error, μ, σ)`
- Use high probability cutoff (e.g., 0.995) for anomaly detection

### 3. Autoencoder + Isolation Forest Ensemble
- Train Isolation Forest on normal data
- Normalize scores and compute weighted ensemble:
  ```
  final_score = 0.6 * AE_score + 0.4 * IF_score
  ```
- Threshold selected from high percentile (e.g., 99.5) of normal samples

---

## Evaluation & Visual Analysis
- Precision, Recall, and F1-score (fraud-focused evaluation)
- Confusion Matrix and heatmap
- ROC Curve and AUC
- Precision–Recall Curve (for imbalance sensitivity)
- Reconstruction error distribution
- Latent-space visualization using PCA

---

## Results
- The autoencoder learns a compact representation of normal transaction behavior
- Reconstruction error provides effective anomaly separation
- Ensemble modeling improves robustness across fraud patterns
- Latent space visualizations show meaningful separation between normal and fraudulent samples

> Note: Performance may vary depending on threshold selection and evaluation strategy.

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn

---

## How to Run
1. Clone the repository
2. Install required dependencies
3. Open the notebook: `DeepAutoEncoder_Fraud_Detection.ipynb`
4. Run all cells sequentially

---

## License
This project is licensed under the **MIT License**.
