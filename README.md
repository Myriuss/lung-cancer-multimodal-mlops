# 🫁 Lung Cancer Detection — Multimodal AI (ML + DL + MLOps)

## Project Overview
This project builds an AI system to predict lung cancer probability using:
- Tabular patient data
- Chest X-ray images

A multimodal approach combining Machine Learning and Deep Learning is deployed via Streamlit.

---

## Objectives
- Build tabular ML model (risk: 0,1,2)
- Build CNN image model
- Build multimodal model
- Deploy using MLOps practices

---

##  Dataset
### Tabular
- Age, Sex
- Smoking history
- Symptoms
- Nodule characteristics

### Images
- Healthy
- Benign
- Malignant

---

##  Models
### Tabular Model
- Logistic Regression
- Random Forest
- Gradient Boosting

### Image Model
- CNN on X-rays

### Multimodal Model
- Combines image + tabular probabilities

---

## Results
| Model | Accuracy |
|------|--------|
| Image only | ~0.51 |
| Multimodal | ~0.83 |

---

##  Project Structure
```
lung-cancer-multimodal-mlops/
│
├── app.py
├── predict.py
├── models/
├── data/
├── notebook.ipynb 
├── train2.ipynb 
├── requirements.txt
└── README.md
```

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

##  Deployment
- Streamlit Cloud : https://lung-cancer-multimodal-mlops-e3aostrulcau5k7np5mvxh.streamlit.app/
- GitHub integration

---

## Tech Stack
- Python
- scikit-learn
- TensorFlow / Keras
- OpenCV
- Streamlit

---

## Limitations
- Small dataset
- Limited generalization

---

##  Improvements
- Transfer learning
- More data
- Better interpretability

