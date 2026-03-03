import streamlit as st
import pandas as pd
import json
from io import BytesIO
from collections import defaultdict, Counter
import subprocess
import sys

# -------------------------
# Installer torch + transformers dynamiquement si absent
# -------------------------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "torch==2.2.0", "transformers==4.41.2"])
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------
# Installer spaCy + modèle dynamiquement si absent
# -------------------------
try:
    import spacy
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "spacy==3.7.4"])
    import spacy

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

nlp = load_spacy()
tokenizer, model = load_sentiment_model()

# -------------------------
# Interface principale
# -------------------------
st.title("Vertex URL Extractor + Analyse de sentiment par marque")
st.markdown(
    "Cette app permet :\n"
    "- D'extraire les URLs Vertex à partir de la colonne `grounding_search_metadata`\n"
    "- D'analyser le sentiment des réponses LLM phrase par phrase pour chaque marque"
)

# Upload fichier
uploaded_file = st.file_uploader("📂 Upload ton fichier Excel", type=["xlsx"])

# Message si aucun fichier
if uploaded_file is None:
    st.info("📌 Veuillez uploader un fichier Excel pour activer les modules.")
else:
    # -------------------------
    # Lecture du fichier
    # -------------------------
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier Excel : {e}")
        st.stop()

    st.success("✅ Fichier chargé avec succès !")

    # -------------------------
    # MODULE 1 – Extraction Vertex URLs
    # -------------------------
    st.header("1️⃣ Vertex URL Extractor")

    col_name = "grounding_search_metadata"
    if col_name not in df.columns:
        st.warning(f"La colonne `{col_name}` n'existe pas dans le fichier.")
    else:

        def extract_vertex_urls(cell):
            if pd.isna(cell):
                return []
            try:
                data = json.loads(cell)
            except:
                return []
            urls = []
            if "sources" in data:
                for source in data["sources"]:
                    if "web" in source and "uri" in source["web"]:
                        url = source["web"]["uri"]
                        if url.startswith("https://vertexaisearch.cloud.google.com"):
                            urls.append(url)
            return urls

        df["extracted_urls"] = df[col_name].apply(extract_vertex_urls)
        df_exploded = df.explode("extracted_urls")
        df_exploded[col_name] = df_exploded["extracted_urls"]
        df_exploded = df_exploded.drop(columns=["extracted_urls"])
        df_exploded = df_exploded.dropna(subset=[col_name])

        st.write("📄 Aperçu des URLs extraites :")
        st.dataframe(df_exploded.head(10))

        # Export Excel
        output_urls = BytesIO()
        df_exploded.to_excel(output_urls, index=False, engine="openpyxl")
        output_urls.seek(0)

        st.download_button(
            label="⬇️ Télécharger le fichier transformé (URLs)",
            data=output_urls,
            file_name="reformatted_All_categ_raw.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # -------------------------
    # MODULE 2 – Analyse sentiment par marque
    # -------------------------
    st.markdown("---")
    st.header("2️⃣ Analyse de sentiment par marque")

    if "G" not in df.columns or "C" not in df.columns:
        st.warning("Les colonnes `G` (réponses LLM) et `C` (thématique) doivent exister pour le module sentiment.")
    else:

        st.info("⚡ Analyse en cours… cela peut prendre quelques minutes selon la taille du fichier.")
        results = []
        progress_bar = st.progress(0)
        total_rows = len(df)

        for idx, row in df.iterrows():
            text = str(row["G"])
            theme = row["C"]
            doc = nlp(text)
            sentences = list(doc.sents)

            for sent in sentences:
                sent_text = sent.text.strip()
                if len(sent_text) < 5:
                    continue
                sent_doc = nlp(sent_text)
                brands = [ent.text.strip() for ent in sent_doc.ents if ent.label_ == "ORG"]
                if not brands:
                    continue

                # Sentiment phrase-level
                inputs = tokenizer(
                    sent_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
                sentiment_score = float(scores[2] - scores[0])

                adjectives = [token.text.lower() for token in sent_doc if token.pos_ == "ADJ"]

                for brand in set(brands):
                    results.append({
                        "brand": brand,
                        "theme": theme,
                        "sentiment": sentiment_score,
                        "adjectives": adjectives
                    })

            progress_bar.progress((idx + 1) / total_rows)

        if not results:
            st.warning("Aucune marque détectée dans les réponses LLM.")
        else:
            results_df = pd.DataFrame(results)
            sentiment_summary = (
                results_df.groupby(["brand", "theme"])["sentiment"].mean().reset_index()
            )

            adj_dict = defaultdict(list)
            for _, row in results_df.iterrows():
                adj_dict[(row["brand"], row["theme"])].extend(row["adjectives"])

            adj_data = []
            for key, adjs in adj_dict.items():
                most_common_adj = [adj for adj, _ in Counter(adjs).most_common(5)]
                adj_data.append({
                    "brand": key[0],
                    "theme": key[1],
                    "top_adjectives": ", ".join(most_common_adj)
                })

            adj_df = pd.DataFrame(adj_data)
            final_df = sentiment_summary.merge(adj_df, on=["brand", "theme"], how="left")
            final_df = final_df.sort_values(by=["theme", "sentiment"], ascending=[True, False])

            st.subheader("📈 Résultats sentiment par marque")
            st.dataframe(final_df)

            output_sentiment = BytesIO()
            final_df.to_excel(output_sentiment, index=False, engine="openpyxl")
            output_sentiment.seek(0)

            st.download_button(
                label="⬇️ Télécharger les résultats sentiment",
                data=output_sentiment,
                file_name="sentiment_analysis_by_brand.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
