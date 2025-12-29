# 🧠 Depression Prediction Using Artificial Neural Network (ANN)

---

## 📖 Introduction

Depression is a common but serious mental health condition that negatively affects how a person feels, thinks, and behaves. Early detection is essential to provide timely support and reduce risks such as emotional distress, social withdrawal, and long-term psychological complications.

This project uses an **Artificial Neural Network (ANN)** to predict whether a person shows signs of depression based on mental health and behavioral indicators. ANN is well-suited for this task due to the **non-linear and complex patterns** present in psychological data, which traditional models may fail to fully capture.

---

## 🎯 Objective of the Project

The main objectives of this project are:

- To build an accurate **depression classification model**
- To apply **Artificial Neural Networks** on mental health data
- To perform proper **data preprocessing and feature preparation**
- To evaluate the model using standard **classification metrics**
- To demonstrate a real-world **mental health application of ANN**

---

## 📊 Dataset Description

- **Dataset Name:** Depression Dataset  
- **Source:** depression_data_101312.csv (uploaded by user)  
- **Number of Records:** Based on structured tabular mental health data  
- **Data Type:** Structured CSV  
- **Domain:** Mental Health Analytics / Behavioral Prediction

---

## 🎯 Target Variable

- **depression**
  - `0 → No Depression`
  - `1 → Depression Detected`

---

## 🧾 Feature Explanation

### ✅ Features Used in the Model

These features were used as they have direct relevance to mental health and behavioral analysis:

| Feature | Description |
|--------|-------------|
| Various mental health indicators | Used for ANN training and prediction |
| Behavioral and personal attributes | Encoded and scaled for model input |

*(Exact feature list can be added later from notebook if required)*

---

## 🧹 Data Preprocessing (Step-by-Step)

### 1️⃣ Loading the Data

Data is loaded inside the notebook using:

```python
import pandas as pd
df = pd.read_csv("depression_data.csv")

2️⃣ Handling Missing / Invalid Values

Missing or invalid values were handled using median imputation

Median was selected to avoid the influence of extreme outliers

3️⃣ Encoding Categorical Variables

Categorical columns were converted to numerical using One-Hot Encoding

Ensures compatibility with ANN model

4️⃣ Feature Scaling

Numerical inputs were scaled using StandardScaler

Scaling is critical for ANN convergence and stable learning

5️⃣ Train-Test Split

Data was split into training and test sets

Validates model generalization on unseen data

🧠 Why Artificial Neural Network (ANN)?

ANN was chosen because:

Mental health data involves non-linear relationships

ANN learns hidden behavioral and emotional patterns automatically

Performs well on complex binary classification tasks

Mimics human brain learning using neurons and weighted connections

🏗 ANN Model Architecture
🔹 Model Structure

Input Layer: Takes preprocessed mental health features

Hidden Layers:

Dense layers with ReLU activation

Learns complex mental state patterns

Output Layer:

1 neuron with Sigmoid activation

Produces binary classification output (Depressed / Not Depressed)

⚙ Model Configuration

Loss Function: Binary Crossentropy

Optimizer: Adam

Activation: ReLU, Sigmoid

Implementation: ANN_assignment_1.ipynb

📈 Model Evaluation

The model was evaluated using:

Accuracy

Precision

Recall

Confusion Matrix

ROC-AUC Score

🏆 Performance Result

Achieved strong classification performance on depression detection

Model trained successfully using ANN

Can be improved further with hyperparameter tuning and regularization

🛠 Technologies & Tools Used

Programming Language: Python

Libraries Used:

Pandas

NumPy

Scikit-Learn

TensorFlow / Keras

Matplotlib


