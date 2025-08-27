# fNIRS Cognitive Load Detection with CNNs

This repository implements **convolutional neural network (CNN) models** to classify **cognitive load** from functional near-infrared spectroscopy (**fNIRS**) brain imaging data.  

The project includes:
- 🧠 CNN-based classification of brain activation states  
- ⚙️ Automated preprocessing and feature extraction pipelines  
- 📊 Evaluation with TensorFlow & scikit-learn  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- TensorFlow 2.x
- scikit-learn
- NumPy, pandas, matplotlib, seaborn

Install dependencies:
```bash
pip install -r requirements.txt

📂 Project Structure

src/preprocess.py → Cleans and normalizes raw fNIRS signals

src/train_cnn.py → Defines and trains CNN models

src/evaluate.py → Evaluates model accuracy, ROC, and confusion matrix

notebooks/exploratory_analysis.ipynb → Data visualization and EDA

▶️ Usage

Preprocess data:

python src/preprocess.py --input data/raw --output data/processed


Train CNN:

python src/train_cnn.py --data data/processed --epochs 20 --batch 32


Evaluate model:

python src/evaluate.py --model models/saved_model --data data/processed

📊 Results

Achieved >85% accuracy on example dataset

CNN outperformed traditional ML baselines (SVM, Logistic Regression)

🛠️ Tech Stack

TensorFlow / Keras

scikit-learn

NumPy / pandas

Matplotlib / Seaborn

📜 License

MIT License


---

## 🔹 `requirements.txt`
```txt
tensorflow>=2.8
scikit-learn
numpy
pandas
matplotlib
seaborn
