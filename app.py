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
# Titre principal
# -------------------------
st.title("Vertex URL Extractor + Analyse de sentiment par marque")

# -------------------------
# 1️⃣ Upload fichier
# -------------------------
uploaded_file = st.file_uploader("Upload ton fichier Excel", type=["xlsx"])

# -------------------------
# MODULE 1 – Extraction Vertex URLs
# -------------------------
if uploaded_file is not None:

    df = pd.read_excel(uploaded_file, sheet_name="All_categ_raw")

    col_name = "grounding_search_metadata"
    if col_name not in df.columns:
        st.error(f"La colonne '{col_name}' n'existe pas.")
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

        output = BytesIO()
        df_exploded.to_excel(output, index=False, engine="openpyxl")
        output.seek(0)

        st.download_button(
            label="Télécharger le fichier transformé",
            data=output,
            file_name="reformatted_All_categ_raw.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# -------------------------
# MODULE 2 – Analyse sentiment par marque
# -------------------------
st.markdown("---")
st.header("📊 Module 2 – Analyse de sentiment par marque (Transformer local)")

if uploaded_file is not None:

    df_sentiment = pd.read_excel(uploaded_file)

    if "G" in df_sentiment.columns and "C" in df_sentiment.columns:

        results = []
        progress_bar = st.progress(0)
        total_rows = len(df_sentiment)

        for idx, row in df_sentiment.iterrows():

            text = str(row["G"])
            theme = row["C"]

            doc = nlp(text)
            sentences = list(doc.sents)

            for sent in sentences:

                sent_text = sent.text.strip()
                if len(sent_text) < 5:
                    continue

                sent_doc = nlp(sent_text)
                # Extraction des marques
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

                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores.detach().numpy()[0]
                sentiment_score = float(scores[2] - scores[0])  # approx -1 à +1

                # Adjectifs phrase-level
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
            st.warning("Aucune marque détectée.")
        else:
            results_df = pd.DataFrame(results)

            # Moyenne sentiment par marque x thématique
            sentiment_summary = (
                results_df
                .groupby(["brand", "theme"])["sentiment"]
                .mean()
                .reset_index()
            )

            # Top adjectifs par marque x thématique
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

            st.subheader("📈 Résultats")
            st.dataframe(final_df)

            # Export Excel
            output_file = "sentiment_analysis_by_brand.xlsx"
            final_df.to_excel(output_file, index=False)

            with open(output_file, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger les résultats",
                    f,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:
        st.error("Les colonnes C (thématique) et G (réponses LLM) doivent exister.")
