import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from transformers import pipeline
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt

# ---------- CONFIGURATION GENERALE ----------
st.set_page_config(
    page_title="IMDB Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ---------- SIDEBAR ----------
st.sidebar.image("https://chloe-rose.fr/wp-content/uploads/2024/01/logo-Openclassrooms.png", width=150)
st.sidebar.title("Dashboard")
st.sidebar.markdown("Projet 9 - OpenClassrooms")

# ---------- CHARGEMENT DES MODELES ET DONNEES ----------
PIPELINE_PATH = "./pipeline_logreg.joblib"
TEST_DATA_PATH = "./results_test.csv"

@st.cache_resource
def load_logreg_pipeline():
    return joblib.load(PIPELINE_PATH)

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)

# Chargement
pipeline_logreg = load_logreg_pipeline()
results_df = load_test_data()
results_df_sample = results_df.sample(2000, random_state=42) # Pour la performance de l'affichage

# Chargement pipeline DeBERTa
deberta = pipeline(
    "text-classification", 
    model="SaidArr/mon-modele-deberta-imdb",
    device=-1  # CPU
)
def query(text):
    return deberta(text)

# Stop words
stop_words = [['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all',
               'am', 'an', 'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be',
               'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but',
               'by', 'can', 'couldn', "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does',
               'doesn', "doesn't", 'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for',
               'from', 'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven',
               "haven't", 'having', 'he', "he'd", "he'll", 'her', 'here', 'hers', 'herself', "he's",
               'him', 'himself', 'his', 'how', 'i', "i'd", 'if', "i'll", "i'm", 'in', 'into', 'is',
               'isn', "isn't", 'it', "it'd", "it'll", "it's", 'its', 'itself', "i've", 'just',
               'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 'most', 'mustn',
               "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 'not',
               'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
               'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's',
               'same', 'shan', "shan't", 'she', "she'd", "she'll", "she's",
               'should', 'shouldn', "shouldn't", "should've", 'so', 'some',
               'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
               'them', 'themselves', 'then', 'there', 'these', 'they', "they'd",
               "they'll", "they're", "they've", 'this', 'those', 'through', 'to',
               'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't",
               'we', "we'd", "we'll", "we're", 'were', 'weren', "weren't", "we've", 'what',
               'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won',
               "won't", 'wouldn', "wouldn't", 'y', 'you', "you'd", "you'll", 'your', "you're",
               'yours', 'yourself', 'yourselves', "you've"]

]


# ---------- TITRE PRINCIPAL ----------
st.title("🎬 Analyse de sentiment - IMDB")

# ---------- TABS PRINCIPAUX ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "Analyse exploratoire", "Prédiction", "Comparaison", "Exemples"
])

# ==========================================
# ======== 1. Analyse exploratoire =========
# ==========================================
with tab1:
        st.header("Statistiques textuelles des avis IMDB")

        st.markdown("Découvrez deux analyses statistiques interactives sur le corpus, ainsi qu'un WordCloud.")

        # --------- 1. Distribution de la longueur des avis (histogramme interactif) ----------
        results_df_sample["nb_mots"] = results_df_sample["review"].apply(lambda x: len(str(x).split()))
        st.subheader("1. Distribution de la taille des avis (en nombre de mots)")
        fig_len = px.histogram(
            results_df_sample,
            x="nb_mots",
            nbins=30,
            color=results_df_sample["target"].map({0: "Négatif", 1: "Positif"}),
            labels={"nb_mots": "Nombre de mots", "color": "Classe"},
            barmode="overlay",
            histnorm="probability density",
            title="Distribution de la taille des avis selon la classe"
        )
        fig_len.update_layout(
            xaxis_title="Nombre de mots par avis",
            yaxis_title="Densité",
            legend_title="Classe"
        )
        st.plotly_chart(fig_len, use_container_width=True)
        
        # --------- 2. Fréquence des mots les plus courants (barplot interactif) ----------
        st.subheader("2. Top 20 des mots les plus fréquents")
        # Nettoyage simple (tu peux améliorer le preprocessing si besoin)
        all_words = " ".join(results_df_sample["review"].astype(str).tolist()).lower().split()
        all_words = [word for word in all_words if word not in stop_words[0] and word.isalpha()]
        freq_dist = Counter(all_words)
        most_common = freq_dist.most_common(20)
        mots, freqs = zip(*most_common)

        fig_freq = px.bar(
            x=mots, y=freqs,
            labels={"x": "Mot", "y": "Fréquence"},
            title="Mots les plus fréquents dans le corpus"
        )
        st.plotly_chart(fig_freq, use_container_width=True)

        # --------- 3. WordCloud (reprend l'existant, possibilité de filtrer par classe) ----------
        st.subheader("3. Nuage de mots")
        choix_wc = st.selectbox(
            "Sélectionnez les avis à afficher dans le WordCloud :",
            ["Tous", "Positifs", "Négatifs"],
            key="wordcloud_stats"
        )
        if choix_wc == "Tous":
            texte = " ".join(results_df_sample['review'].astype(str).tolist())
        elif choix_wc == "Positifs":
            texte = " ".join(results_df_sample[results_df_sample['target'] == 1]['review'].astype(str).tolist())
        else:
            texte = " ".join(results_df_sample[results_df_sample['target'] == 0]['review'].astype(str).tolist())

        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(texte)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# =======================================
# =========== 2. Prédiction ============
# =======================================
with tab2:
    st.header("Prédiction individuelle")
    choix = st.radio(
        "Choisissez le mode de prédiction :",
        ("DeBERTa-v3", "Régression Logistique", "Comparer les deux modèles")
    )
    user_input = st.text_area("Saisissez un avis à analyser...")

    if st.button("Analyser", key="predict_btn"):
        if not user_input.strip():
            st.warning("Veuillez entrer un texte à analyser.")
        else:
            if choix == "DeBERTa-v3":
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

            else:  # Comparer les deux modèles
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
# =========== 3. Comparaison ============
# =======================================
with tab3:
    st.header("Comparaison des modèles sur le jeu de test")

    # Extraction des valeurs
    y_true = results_df["target"].values
    y_pred_logreg = results_df["y_pred_logreg"].values
    proba_logreg = results_df["proba_logreg"].values
    y_pred_deberta = results_df["y_pred_deberta"].values
    proba_deberta = results_df["proba_deberta"].values

    # Scores globaux
    df_scores = pd.DataFrame({
        "Accuracy": [
            accuracy_score(y_true, y_pred_logreg),
            accuracy_score(y_true, y_pred_deberta)
        ],
        "F1-score": [
            f1_score(y_true, y_pred_logreg),
            f1_score(y_true, y_pred_deberta)
        ],
        "Precision": [
            precision_score(y_true, y_pred_logreg),
            precision_score(y_true, y_pred_deberta)
        ],
        "Recall": [
            recall_score(y_true, y_pred_logreg),
            recall_score(y_true, y_pred_deberta)
        ]
    }, index=["Logreg+TFIDF", "DeBERTa-v3"])

    # Tableau comparatif
    st.subheader("Tableau comparatif des scores")
    st.dataframe(df_scores.style.format("{:.3f}"))

    # Barplot interactif (Plotly)
    st.subheader("Barplot des performances (interactif)")
    fig_bar = go.Figure()
    for metric in df_scores.columns:
        fig_bar.add_trace(go.Bar(
            x=df_scores.index,
            y=df_scores[metric],
            name=metric
        ))
    fig_bar.update_layout(
        barmode='group',
        xaxis_title="Modèle",
        yaxis_title="Score",
        legend_title="Métrique"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    

    # Matrice de confusion (Plotly heatmap)
    st.subheader("Matrices de confusion (interactif)")
    col1, col2 = st.columns(2)
    with col1:
        cm = confusion_matrix(y_true, y_pred_logreg)
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=['Négatif', 'Positif'], y=['Négatif', 'Positif'],
            colorscale='Blues'
        )
        fig_cm.update_layout(title="LogReg")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        cm2 = confusion_matrix(y_true, y_pred_deberta)
        fig_cm2 = ff.create_annotated_heatmap(
            z=cm2, x=['Négatif', 'Positif'], y=['Négatif', 'Positif'],
            colorscale='Greens'
        )
        fig_cm2.update_layout(title="DeBERTa-v3")
        st.plotly_chart(fig_cm2, use_container_width=True)


    # Courbe ROC (Plotly)
    st.subheader("Courbe ROC (interactif)")
    fpr_logreg, tpr_logreg, _ = roc_curve(y_true, proba_logreg)
    fpr_deberta, tpr_deberta, _ = roc_curve(y_true, proba_deberta)
    auc_logreg = auc(fpr_logreg, tpr_logreg)
    auc_deberta = auc(fpr_deberta, tpr_deberta)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_logreg, y=tpr_logreg, mode='lines', name=f'Logreg (AUC={auc_logreg:.2f})'
    ))
    fig_roc.add_trace(go.Scatter(
        x=fpr_deberta, y=tpr_deberta, mode='lines', name=f'DeBERTa-v3 (AUC={auc_deberta:.2f})'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')
    ))
    fig_roc.update_layout(
        title="Courbe ROC des modèles",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs"
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# =======================================
# ============= 4. Exemples =============
# =======================================
with tab4:
    st.header("Explorer des exemples selon la performance des deux modèles")
    choix = st.selectbox(
        "Quel type d'exemples voulez-vous afficher ?",
        [
            "Bien classés par les deux",
            "Bien classés par DeBERTa-v3 seulement",
            "Bien classés par LogReg seulement",
            "Mal classés par les deux"
        ]
    )
    pred_deberta = results_df["y_pred_deberta"]
    pred_logreg = results_df["y_pred_logreg"]
    true_label = results_df["target"]

    # Création des masques pour chaque cas
    if choix == "Bien classés par les deux":
        mask = (pred_deberta == true_label) & (pred_logreg == true_label)
    elif choix == "Bien classés par DeBERTa-v3 seulement":
        mask = (pred_deberta == true_label) & (pred_logreg != true_label)
    elif choix == "Bien classés par LogReg seulement":
        mask = (pred_logreg == true_label) & (pred_deberta != true_label)
    else:
        mask = (pred_deberta != true_label) & (pred_logreg != true_label)

    exemples = results_df[mask]
    n = st.slider("Combien d'exemples afficher ?", min_value=1, max_value=5, value=2)

    if len(exemples) == 0:
        st.info("Aucun exemple ne correspond à ce critère dans votre jeu de test.")
    else:
        exemples = exemples.sample(n=min(n, len(exemples)), random_state=42)
        for i, row in exemples.iterrows():
            conf_deberta = row['proba_deberta'] if row['y_pred_deberta'] == 1 else 1 - row['proba_deberta']
            conf_logreg = row['proba_logreg'] if row['y_pred_logreg'] == 1 else 1 - row['proba_logreg']
            st.write(f"**Texte :** {row['review']}")
            st.write(f"Vrai label : {'Positif' if row['target'] == 1 else 'Négatif'}")
            st.write(f"DeBERTa-v3 : {'Positif' if row['y_pred_deberta'] == 1 else 'Négatif'} (confiance : {conf_deberta*100:.2f}%)")
            st.write(f"LogReg : {'Positif' if row['y_pred_logreg'] == 1 else 'Négatif'} (confiance : {conf_logreg*100:.2f}%)")
            st.markdown("---")
            

# =======================================
# ============ FOOTER ===================
# =======================================
st.markdown("---")
st.caption("Projet 9 - OpenClassrooms : Développez une preuve de concept | Dashboard réalisé avec Streamlit | Said Arrazouaki")
