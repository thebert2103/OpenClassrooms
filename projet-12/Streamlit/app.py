import streamlit as st
import pandas as pd
import pickle

# Titre de l'app
st.title("💶 Authentification des billets 💶")

# Chargement du modèle
with open("model_logreg.pkl", "rb") as file:
    model = pickle.load(file)

# Upload du fichier CSV
uploaded_file = st.file_uploader("Chargez votre fichier CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Aperçu des données
    st.subheader("Aperçu des données")
    st.write(data)

    # Features utilisées
    expected_features = ["margin_up", "height_right", "height_left", "length", "diagonal", "margin_low"]
    X_new = data[expected_features]

    # Prédiction des classes
    predictions = model.predict(X_new)

    # Prédiction des probabilités pour la classe positive (vrai billet)
    proba = model.predict_proba(X_new)[:, 1]

    # Ajouter la colonne de prédiction (booléens)
    data["log_pred"] = predictions

    # Ajouter la colonne des probabilités, arrondie à 3 décimales
    data["probabilité"] = proba.round(3)

    # Légende pour les symboles
    st.subheader("Résultat des prédictions")
    st.markdown("**Légende :** 🟢 = vrai billet &nbsp;&nbsp;&nbsp; 🔴 = faux billet")

    # Création d'une copie pour affichage avec les ronds
    display_df = data.copy()
    display_df["log_pred"] = display_df["log_pred"].map({True: "🟢", False: "🔴"})

    # Affichage stylisé avec 2 décimales
    st.dataframe(display_df.style.format(precision=2))

    # Export avec les données originales (True/False et probabilité)
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les résultats", data=csv, file_name="resultats_predictions.csv", mime="text/csv")