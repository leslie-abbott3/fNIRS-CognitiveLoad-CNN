# fNIRS Cognitive Load Detection with Deep Learning

This repository implements **convolutional neural networks (CNNs)** and classical ML models for classifying **cognitive load** from functional near-infrared spectroscopy (**fNIRS**) brain imaging data.

Includes:
- Automated preprocessing: filtering, normalization, sliding windows
- Synthetic data generator for reproducibility
- CNN with dropout, batch norm, TensorBoard logging
- Baseline models (SVM, Random Forest) for comparison
- Full evaluation pipeline (accuracy, F1, ROC-AUC, confusion matrix)

---

## 🚀 Quick Start

### Install requirements
```bash
pip install -r requirements.txt
Generate synthetic dataset
bash
Copy code
python src/data_utils.py --generate --samples 200 --channels 16 --length 200
Preprocess data
bash
Copy code
python src/preprocess.py --input data/raw --output data/processed
Train CNN
bash
Copy code
python src/train.py --config config/config.yaml
Evaluate
bash
Copy code
python src/evaluate.py --model models/saved_model/cnn_model.h5 --data data/processed
📊 Results
On synthetic data:

CNN accuracy ~90%

Outperforms SVM and Random Forest baselines

🛠️ Tech Stack
TensorFlow / Keras

scikit-learn

NumPy / pandas

Matplotlib / Seaborn

PyYAML (config management)

📜 License
MIT License

yaml
Copy code

---

## 🔹 `requirements.txt`
```txt
tensorflow>=2.8
scikit-learn
numpy
pandas
matplotlib
seaborn
pyyaml
scipy
