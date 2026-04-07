import numpy as np
import cv2
import joblib
import tensorflow as tf

IMG_SIZE = 128

TAB_FEATURES = ["age", "sexe_masculin"]

def load_and_preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Impossible de lire l'image téléchargée.")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def load_models():
    tab_model = joblib.load("models/model_tabulaire.pkl")
    scaler_tab = joblib.load("models/scaler_tab.pkl")
    image_model = tf.keras.models.load_model("models/cnn_image.keras")
    multimodal_model = tf.keras.models.load_model("models/multimodal_clean.keras")
    return tab_model, scaler_tab, image_model, multimodal_model

def predict_all(age, sexe_masculin, image_file):
    tab_model, scaler_tab, image_model, multimodal_model = load_models()

    # Tabulaire propre pour le multimodal
    X_tab = np.array([[age, sexe_masculin]], dtype="float32")
    X_tab_scaled = scaler_tab.transform(X_tab).astype("float32")

    # Image
    X_img = load_and_preprocess_image(image_file)

    # Prédiction image seule
    img_prob = float(image_model.predict(X_img, verbose=0)[0][0])
    img_pred = 1 if img_prob >= 0.5 else 0

    # Prédiction multimodale propre
    multi_prob = float(multimodal_model.predict([X_img, X_tab_scaled], verbose=0)[0][0])
    multi_pred = 1 if multi_prob >= 0.5 else 0

    return {
        "image_seule": {
            "prediction": img_pred,
            "probabilite_cancer": round(img_prob, 4)
        },
        "multimodal_propre": {
            "prediction": multi_pred,
            "probabilite_cancer": round(multi_prob, 4)
        }
    }