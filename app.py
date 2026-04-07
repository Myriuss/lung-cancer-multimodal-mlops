import streamlit as st
from PIL import Image
from predict import predict_all

st.set_page_config(
    page_title="Détection du cancer pulmonaire",
    page_icon="🫁",
    layout="wide"
)

# -------------------------
# Style
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3, h4 {
    color: #1f2937;
}
.card {
    background-color: white;
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.result-good {
    background-color: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 1rem;
    border-radius: 12px;
}
.result-warn {
    background-color: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 1rem;
    border-radius: 12px;
}
.result-bad {
    background-color: #fef2f2;
    border-left: 6px solid #ef4444;
    padding: 1rem;
    border-radius: 12px;
}
.small-text {
    color: #6b7280;
    font-size: 0.95rem;
}
.metric-label {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Helpers
# -------------------------
def format_binary_prediction(pred: int) -> str:
    return "Cancer probable" if pred == 1 else "Cancer non probable"

def result_box(title, prediction, probability):
    prob_percent = probability * 100

    if 45 <= prob_percent <= 55:
        css_class = "result-warn"
        interpretation = "Prédiction incertaine"
    elif prediction == 1:
        css_class = "result-bad"
        interpretation = "Risque positif détecté"
    else:
        css_class = "result-good"
        interpretation = "Risque négatif détecté"

    st.markdown(
        f"""
        <div class="{css_class}">
            <h4 style="margin-bottom:0.3rem;">{title}</h4>
            <p style="font-size:1.05rem; margin:0.2rem 0;"><b>{format_binary_prediction(prediction)}</b></p>
            <p style="margin:0.2rem 0;">Probabilité : <b>{probability:.4f}</b> ({prob_percent:.1f}%)</p>
            <p class="small-text" style="margin-top:0.4rem;">{interpretation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="card">
    <h1>🫁 Détection du cancer pulmonaire</h1>
    <p class="small-text">
        Cette application combine des données cliniques patient et une radio thoracique
        pour estimer le risque de malignité et la probabilité d’un cancer pulmonaire.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Layout principal
# -------------------------
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Informations patient")

    age = st.slider("Âge", min_value=18, max_value=100, value=50)

    sexe = st.radio("Sexe", ["Femme", "Homme"], horizontal=True)
    sexe_masculin = 1 if sexe == "Homme" else 0

    st.markdown("### Données cliniques")

    col1, col2 = st.columns(2)

    with col1:
        presence_nodule = 1 if st.toggle("Présence d’un nodule") else 0
        subtilite_nodule = st.slider("Subtilité du nodule", min_value=1, max_value=5, value=3)
        x_nodule_norm = st.slider("Position X du nodule", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        y_nodule_norm = st.slider("Position Y du nodule", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        tabagisme_paquets_annee = st.slider("Tabagisme (paquets/année)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)

    with col2:
        toux_chronique = 1 if st.toggle("Toux chronique") else 0
        dyspnee = 1 if st.toggle("Dyspnée") else 0
        douleur_thoracique = 1 if st.toggle("Douleur thoracique") else 0
        perte_poids = 1 if st.toggle("Perte de poids") else 0
        spo2 = st.slider("SpO2", min_value=80, max_value=100, value=98)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Radio thoracique")
    uploaded_file = st.file_uploader(
        "Télécharger une image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Image chargée", use_container_width=True)
    else:
        st.info("Ajoute une radio thoracique pour lancer la prédiction.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Bouton principal
# -------------------------
st.markdown("###")
launch = st.button("🔍 Lancer la prédiction", use_container_width=True)

# -------------------------
# Prédiction
# -------------------------
if launch:
    if uploaded_file is None:
        st.error("Veuillez charger une radio thoracique.")
    else:
        try:
            uploaded_file.seek(0)

            patient_data = {
                "age": age,
                "sexe_masculin": sexe_masculin,
                "presence_nodule": presence_nodule,
                "subtilite_nodule": subtilite_nodule,
                "x_nodule_norm": x_nodule_norm,
                "y_nodule_norm": y_nodule_norm,
                "tabagisme_paquets_annee": tabagisme_paquets_annee,
                "toux_chronique": toux_chronique,
                "dyspnee": dyspnee,
                "douleur_thoracique": douleur_thoracique,
                "perte_poids": perte_poids,
                "spo2": spo2,
            }

            results = predict_all(patient_data, uploaded_file)

            st.markdown("## Résultats")

            col_res1, col_res2 = st.columns(2)

            with col_res1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Modèle tabulaire")

                probas = results["tabulaire"]["probas"][0]
                classes = ["Faible", "Intermédiaire", "Élevé"]
                best_idx = max(range(3), key=lambda i: probas[i])

                st.write(f"**Risque de malignité prédit :** {classes[best_idx]}")
                st.write("**Probabilités par classe :**")

                st.write(f"Faible : {probas[0]:.4f}")
                st.progress(float(probas[0]))

                st.write(f"Intermédiaire : {probas[1]:.4f}")
                st.progress(float(probas[1]))

                st.write(f"Élevé : {probas[2]:.4f}")
                st.progress(float(probas[2]))

                st.markdown('</div>', unsafe_allow_html=True)

            with col_res2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Décision finale")

                result_box(
                    "Modèle image seule",
                    results["image_seule"]["prediction"],
                    results["image_seule"]["probabilite"]
                )

                result_box(
                    "Modèle multimodal",
                    results["multimodal"]["prediction"],
                    results["multimodal"]["probabilite"]
                )

                st.markdown('</div>', unsafe_allow_html=True)

            st.info(
                "Le modèle multimodal combine la radio thoracique et les probabilités produites par le modèle tabulaire."
            )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")