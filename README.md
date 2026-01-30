# Deep Autoencoder Credit Card Fraud Detection

## Description
This project implements an **unsupervised credit card fraud detection system** using a **Deep Autoencoder neural network**.  
The model is trained exclusively on legitimate transactions and identifies potential fraud by analyzing **reconstruction error**, making it suitable for **highly imbalanced datasets**.

The project also explores **probabilistic anomaly scoring** and an **Autoencoder + Isolation Forest ensemble** approach to improve detection robustness.

---

## Dataset
The model is developed using a **publicly available credit card transactions dataset** containing anonymized features derived via PCA.  
The dataset is not included in this repository.

---

## Approach
1. Preprocess transaction data and scale monetary values
2. Train a deep autoencoder on non-fraudulent transactions
3. Compute reconstruction error on unseen data
4. Detect fraud using:
   - Reconstruction error thresholding
   - Probabilistic anomaly scoring
   - Autoencoder + Isolation Forest ensemble
5. Evaluate results using classification metrics and visual analysis

---

## Model Architecture
- Fully connected Deep Autoencoder
- Encoder layers: 64 → 32 → 16 → 8
- Decoder layers: 8 → 16 → 32 → 64 → Input
- Batch Normalization and LeakyReLU activations
- Dropout for regularization
- Loss function: Log-Cosh
- Optimizer: Adam

---

## Evaluation
Model evaluation includes:
- Precision, Recall, and F1-score
- Confusion Matrix
- ROC Curve and AUC
- Precision–Recall Curve
- Reconstruction error distribution
- Latent space visualization using PCA

Labels are used strictly for evaluation and not during training.

---

## Results
- The autoencoder learns a compact representation of normal transaction behavior
- Reconstruction error enables effective anomaly detection
- Ensemble modeling improves robustness across fraud patterns
- Latent space visualization demonstrates meaningful separation

Performance may vary depending on threshold selection.

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy

---

## How to Run
1. Clone the repository
2. Install required dependencies
3. Open the notebook: DeepAutoEncoder_Fraud_Detection.ipynb
4. Run all cells sequentially

## License
This project is licensed under the MIT License.
