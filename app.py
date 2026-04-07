import streamlit as st
from PIL import Image
from predict import predict_all

st.set_page_config(page_title="Détection du cancer pulmonaire", layout="centered")

st.title("Détection du cancer pulmonaire")
st.write("Application de prédiction basée sur une radio thoracique et des données patient.")

st.subheader("Informations patient")
age = st.number_input("Âge", min_value=1, max_value=120, value=50)
sexe = st.selectbox("Sexe", ["Femme", "Homme"])
sexe_masculin = 1 if sexe == "Homme" else 0

st.subheader("Radio thoracique")
uploaded_file = st.file_uploader("Télécharger une image JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Image téléchargée", use_container_width=True)

if st.button("Lancer la prédiction"):
    if uploaded_file is None:
        st.error("Veuillez charger une radio thoracique.")
    else:
        try:
            uploaded_file.seek(0)
            results = predict_all(age, sexe_masculin, uploaded_file)

            st.success("Prédiction effectuée avec succès")

            st.subheader("Résultat - Modèle image seule")
            pred_img = results["image_seule"]["prediction"]
            prob_img = results["image_seule"]["probabilite_cancer"]
            st.write(f"Prédiction : {'Cancer probable' if pred_img == 1 else 'Cancer non probable'}")
            st.write(f"Probabilité : {prob_img}")

            st.subheader("Résultat - Modèle multimodal propre")
            pred_multi = results["multimodal_propre"]["prediction"]
            prob_multi = results["multimodal_propre"]["probabilite_cancer"]
            st.write(f"Prédiction : {'Cancer probable' if pred_multi == 1 else 'Cancer non probable'}")
            st.write(f"Probabilité : {prob_multi}")

            st.info("Le modèle multimodal combine l’image et des variables cliniques non fuyantes.")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")