import streamlit as st
import os
import pandas as pd
from utilis import preprocess_data, train_models, save_model

# Configuration de la page
st.set_page_config(page_title="ML App", layout="wide", page_icon="🧠")

# Appliquer des styles personnalisés
st.markdown("""
    <style>
    .main {
        background-color: #f4f4f9;
    }
    .sidebar .sidebar-content {
        background-color: #2a3d66;
    }
    .block-container {
        padding: 20px;
    }
    h1 {
        color: #2a3d66;
    }
    .stButton>button {
        background-color: #ff6347;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px;
    }
    .stSelectbox>div>div>input {
        color: #2a3d66;
    }
    </style>
""", unsafe_allow_html=True)

# Sélection de la page
page = st.sidebar.selectbox("📂 Choisissez une page", ["Page 1: Téléchargement des données", "Page 2: Entraînement et Prédiction"])

# Page 1 : Téléchargement des données
if page == "Page 1: Téléchargement des données":
    st.title("📊 Page 1: Téléchargement des données")

    uploaded_file = st.file_uploader("📁 Téléchargez un fichier CSV", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)

            with open(f"data/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success("Fichier chargé avec succès.")
            st.subheader("Aperçu des données")
            st.dataframe(data.head())

            # Stocker les données dans la session pour la page 2
            st.session_state["data"] = data

            st.info("Passons maintenant à l'entraînement des modèles sur la Page 2.")

        except Exception as e:
            st.error(f"Impossible de lire le fichier CSV : {e}")

# Page 2 : Entraînement des modèles et Prédiction
if page == "Page 2: Entraînement et Prédiction":
    st.title("🧠 Page 2: Entraînement des Modèles et Prédiction")

    # Vérifier si les données sont déjà chargées depuis la page 1
    if "data" not in st.session_state:
        st.error("Veuillez d'abord télécharger un fichier CSV sur la Page 1.")
    else:
        data = st.session_state["data"]
        task_type = st.selectbox("🔍 Type de tâche", ["Classification", "Régression"], index=0)

        # Sélection de la variable cible
        target_variable = st.selectbox("🎯 Variable cible", data.columns)

        if st.button("🚀 Entraîner les modèles"):
            try:
                with st.spinner("Prétraitement et entraînement..."):
                    X, y = preprocess_data(data, target_variable, task_type)
                    models, metrics = train_models(X, y, task_type)
                    save_model(models, f"{task_type}_model")

                st.success("Modèles entraînés avec succès !")
                st.subheader("📊 Résultats des modèles")
                st.dataframe(pd.DataFrame(metrics).T)

                # Sauvegarder les modèles et caractéristiques dans la session
                st.session_state["models"] = models
                st.session_state["features"] = list(X.columns)
                st.session_state["task_type"] = task_type

            except ValueError as ve:
                st.error(f"Erreur de traitement : {ve}")
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")

        if "models" in st.session_state:
            st.subheader("🔮 Faire une prédiction")
            model_name = st.selectbox("🧠 Choisissez un modèle", list(st.session_state["models"].keys()))

            # Entrée des nouvelles données
            inputs = []
            for feature in st.session_state["features"]:
                val = st.text_input(f"{feature}", "0")
                try:
                    inputs.append(float(val))
                except ValueError:
                    st.error(f"Entrée invalide pour {feature} : '{val}' n'est pas un nombre.")

            # Prédiction
            if st.button("Prédire"):
                try:
                    selected_model = st.session_state["models"][model_name]
                    prediction = selected_model.predict([inputs])
                    st.success(f"Résultat de la prédiction : {prediction[0]}")
                except Exception as e:
                    st.error(f"Erreur pendant la prédiction : {e}")
