import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import numpy as np
import pandas as pd
# from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline

deberta = pipeline(
    "text-classification", 
    model="SaidArr/mon-modele-deberta-imdb",
    device=-1  # -1 = cpu, 0 = gpu si dispo
)

def query(text):
    result = deberta(text)
    return result

# --- Chemins locaux ---

PIPELINE_PATH = "./pipeline_logreg.joblib"
TEST_DATA_PATH = "./results_test.csv"


@st.cache_resource
def load_logreg_pipeline():
    pipeline = joblib.load(PIPELINE_PATH)
    return pipeline

@st.cache_data
def load_test_data():
    df = pd.read_csv(TEST_DATA_PATH)
    return df

pipeline_logreg = load_logreg_pipeline()
results_df = load_test_data()

st.title("Analyse de sentiment sur les avis IMDB")

# Onglet supplémentaire Comparaison
tab1, tab2, tab3 = st.tabs(["Prédiction", "Comparaison", "Exemples"])

# =======================================
# ==== Onglet 1 : Prédiction ============
# =======================================
with tab1:
    st.header("Prédiction individuelle")

    choix = st.radio(
        "Choisissez le mode de prédiction :",
        ("DeBERTa-v3", "Régression Logistique", "Comparer les deux modèles"),
        key="radio_predict"
    )
    user_input = st.text_area("Saisissez un avis à analyser...", key="predict_input")

    if st.button("Analyser", key="predict_btn"):
        if not user_input.strip():
            st.warning("Veuillez entrer un texte à analyser.")
        else:
            if choix == "DeBERTa-v3":
                # Prédiction DeBERTa-v3
                results = query(user_input)

                pred_class = results[0]['label']
                proba = results[0]['score']

                st.success(f"**Résultat :** {'Positif' if pred_class == 'Label_1' else 'Négatif'}")
                st.write(f"Confiance : {proba*100:.2f} %")
                st.caption("Modèle utilisé : DeBERTa-v3")

            elif choix == "Régression Logistique":
                pred_class = pipeline_logreg.predict([user_input])[0]
                proba = pipeline_logreg.predict_proba([user_input])[0]
                st.success(f"**Résultat :** {'Positif' if pred_class == 1 else 'Négatif'}")
                st.write(f"Confiance : {np.max(proba).item()*100:.2f} %")
                st.caption("Modèle utilisé : Régression logistique")

            elif choix == "Comparer les deux modèles":
                colA, colB = st.columns(2)
                with colA:
                    st.subheader("DeBERTa-v3")
                    results = query(user_input)

                    pred_class = results[0]['label']
                    proba = results[0]['score']
                    st.write(f"**Résultat :** {'Positif' if pred_class == 'Label_1' else 'Négatif'}")
                    st.write(f"Confiance : {proba*100:.2f} %")
                    st.caption("Modèle : DeBERTa-v3")

                with colB:
                    st.subheader("Régression Logistique")
                    pred_class_lr = pipeline_logreg.predict([user_input])[0]
                    proba_lr = pipeline_logreg.predict_proba([user_input])[0]
                    st.write(f"**Résultat :** {'Positif' if pred_class_lr == 1 else 'Négatif'}")
                    st.write(f"Confiance : {np.max(proba_lr).item()*100:.2f} %")
                    st.caption("Modèle : Régression Logistique")




# =======================================
# ==== Onglet 2 : Comparaison =========
# =======================================
with tab2:
    st.header("Comparaison des modèles sur le jeu de test")

    # Récupérer les colonnes nécessaires
    y_true = results_df["target"].values
    y_pred_logreg = results_df["y_pred_logreg"].values
    proba_logreg = results_df["proba_logreg"].values
    y_pred_deberta = results_df["y_pred_deberta"].values
    proba_deberta = results_df["proba_deberta"].values

    # Calcul des scores globaux pour chaque modèle

    # Baseline logreg
    acc_logreg = accuracy_score(y_true, y_pred_logreg)
    f1_logreg = f1_score(y_true, y_pred_logreg)
    prec_logreg = precision_score(y_true, y_pred_logreg)
    rec_logreg = recall_score(y_true, y_pred_logreg)

    # DeBERTa
    acc_deberta = accuracy_score(y_true, y_pred_deberta)
    f1_deberta = f1_score(y_true, y_pred_deberta)
    prec_deberta = precision_score(y_true, y_pred_deberta)
    rec_deberta = recall_score(y_true, y_pred_deberta)

    df_scores = pd.DataFrame({
        "Accuracy": [acc_logreg, acc_deberta],
        "F1-score": [f1_logreg, f1_deberta],
        "Precision": [prec_logreg, prec_deberta],
        "Recall": [rec_logreg, rec_deberta]
    }, index=["Logreg+TFIDF", "DeBERTa-v3"])

    st.subheader("Tableau comparatif des scores")
    st.dataframe(df_scores.style.format("{:.3f}"))

    # Barplot des scores
    st.subheader("Barplot des performances")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax = df_scores.plot.bar(
        rot=0, 
        color=["#59bb18", "#a0a04e", "#069e8f", "#07076d"]
    )

    # Déplacer la légende à droite, en dehors du graphique
    ax.legend(
        bbox_to_anchor=(1.05, 1),   # positionne la légende à droite
        loc='upper left', 
        borderaxespad=0.
    )

    st.pyplot(ax.get_figure(), use_container_width=True)
    plt.clf()


    # Matrices de confusion
    st.subheader("Matrices de confusion")
    cm_logreg = confusion_matrix(y_true, y_pred_logreg)
    cm_deberta = confusion_matrix(y_true, y_pred_deberta)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Régression Logistique")
        disp1 = ConfusionMatrixDisplay(cm_logreg)
        fig1, ax1 = plt.subplots()
        disp1.plot(ax=ax1)
        st.pyplot(fig1)
        plt.clf()
    with col2:
        st.write("DeBERTa-v3")
        disp2 = ConfusionMatrixDisplay(cm_deberta)
        fig2, ax2 = plt.subplots()
        disp2.plot(ax=ax2)
        st.pyplot(fig2)
        plt.clf()

    # Courbe ROC
    st.subheader("Courbe ROC (Comparaison)")
    fpr_logreg, tpr_logreg, _ = roc_curve(y_true, proba_logreg)
    fpr_deberta, tpr_deberta, _ = roc_curve(y_true, proba_deberta)
    auc_logreg = auc(fpr_logreg, tpr_logreg)
    auc_deberta = auc(fpr_deberta, tpr_deberta)

    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr_logreg, tpr_logreg, label=f"Logreg (AUC={auc_logreg:.2f})")
    ax_roc.plot(fpr_deberta, tpr_deberta, label=f"DeBERTa-v3 (AUC={auc_deberta:.2f})")
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_xlabel("Taux de faux positifs")
    ax_roc.set_ylabel("Taux de vrais positifs")
    ax_roc.set_title("Courbe ROC")

    # Légende à droite en dehors du graphique
    ax_roc.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )

    st.pyplot(fig_roc)
    plt.clf()


# =======================================
# ==== Onglet 3 : Exemples ==============
# =======================================
with tab3:
    st.header("Explorer des exemples selon la performance des deux modèles")

    # Choix du type d'exemple
    choix = st.selectbox(
        "Quel type d'exemples voulez-vous afficher ?",
        [
            "Bien classés par les deux",
            "Bien classés par DeBERTa-v3 seulement",
            "Bien classés par LogReg seulement",
            "Mal classés par les deux"
        ],
        key="type_example"
    )

    # Création des masques pour chaque cas
    pred_deberta = results_df["y_pred_deberta"]
    pred_logreg = results_df["y_pred_logreg"]
    true_label = results_df["target"]

    # Cas 1 : bien classés par les deux
    if choix == "Bien classés par les deux":
        mask = (pred_deberta == true_label) & (pred_logreg == true_label)
    # Cas 2 : bien classés par DeBERTa seulement
    elif choix == "Bien classés par DeBERTa-v3 seulement":
        mask = (pred_deberta == true_label) & (pred_logreg != true_label)
    # Cas 3 : bien classés par LogReg seulement
    elif choix == "Bien classés par LogReg seulement":
        mask = (pred_logreg == true_label) & (pred_deberta != true_label)
    # Cas 4 : mal classés par les deux
    else:
        mask = (pred_deberta != true_label) & (pred_logreg != true_label)

    exemples = results_df[mask]
    n = st.slider("Combien d'exemples afficher ?", min_value=1, max_value=5, value=2)

    if len(exemples) == 0:
        st.info("Aucun exemple ne correspond à ce critère dans votre jeu de test.")
    else:
        # Afficher n exemples aléatoires du sous-ensemble
        exemples = exemples.sample(n=min(n, len(exemples)), random_state=42)
        for i, row in exemples.iterrows():
            if row['y_pred_deberta'] == 1:
                conf_deberta = row['proba_deberta']
            else:
                conf_deberta = 1 - row['proba_deberta']
            if row['y_pred_logreg'] == 1:
                conf_logreg = row['proba_logreg']
            else:
                conf_logreg = 1 - row['proba_logreg']
            st.write(f"**Texte :** {row['review']}")
            st.write(f"Vrai label : {'Positif' if row['target'] == 1 else 'Négatif'}")
            st.write(f"DeBERTa-v3 : {'Positif' if row['y_pred_deberta'] == 1 else 'Négatif'} (confiance : {conf_deberta*100:.2f}%)")
            st.write(f"LogReg : {'Positif' if row['y_pred_logreg'] == 1 else 'Négatif'} (confiance : {conf_logreg*100:.2f}%)")

            st.markdown("---")

st.markdown("---")
st.caption("Projet 9 - OpenClassrooms : Analyse de sentiment sur les avis IMDB")