import numpy as np
import cv2
import joblib
import tensorflow as tf
from pathlib import Path

IMG_SIZE = 128

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

TAB_FEATURES = [
    "age",
    "sexe_masculin",
    "presence_nodule",
    "subtilite_nodule",
    "x_nodule_norm",
    "y_nodule_norm",
    "tabagisme_paquets_annee",
    "toux_chronique",
    "dyspnee",
    "douleur_thoracique",
    "perte_poids",
    "spo2"
]

# -------------------------
# Load models
# -------------------------
def load_models():
    tab_model = joblib.load(MODELS_DIR / "model_tabulaire.pkl")
    scaler_tab = joblib.load(MODELS_DIR / "scaler_tab.pkl")
    image_model = tf.keras.models.load_model(MODELS_DIR / "cnn_image.keras")
    multimodal_model = tf.keras.models.load_model(MODELS_DIR / "multimodal_clean.keras")
    return tab_model, scaler_tab, image_model, multimodal_model

# -------------------------
# Image preprocessing
# -------------------------
def load_and_preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Image invalide")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# -------------------------
# Build tabular input
# -------------------------
def build_tab_array(patient_data):
    return np.array([[patient_data[col] for col in TAB_FEATURES]], dtype="float32")

# -------------------------
# Main prediction
# -------------------------
def predict_all(patient_data, image_file):
    tab_model, scaler_tab, image_model, multimodal_model = load_models()

    # tabulaire
    X_tab = build_tab_array(patient_data)
    X_tab_scaled = scaler_tab.transform(X_tab)

    # image
    X_img = load_and_preprocess_image(image_file)

    # modèle 1 → probabilités
    tab_proba = tab_model.predict_proba(X_tab_scaled)

    # modèle image seule
    img_prob = float(image_model.predict(X_img)[0][0])
    img_pred = 1 if img_prob >= 0.5 else 0

    # modèle multimodal
    multi_prob = float(multimodal_model.predict([X_img, tab_proba])[0][0])
    multi_pred = 1 if multi_prob >= 0.5 else 0

    return {
        "tabulaire": {
            "probas": tab_proba.tolist()
        },
        "image_seule": {
            "prediction": img_pred,
            "probabilite": round(img_prob, 4)
        },
        "multimodal": {
            "prediction": multi_pred,
            "probabilite": round(multi_prob, 4)
        }
    }