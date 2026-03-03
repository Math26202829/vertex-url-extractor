import streamlit as st
import pandas as pd
import json
from io import BytesIO

# 📌 Titre simple
st.title("Vertex URL Extractor")

# 📌 1. Upload du fichier
uploaded_file = st.file_uploader("Upload ton fichier Excel", type=["xlsx"])

if uploaded_file is not None:

    # 📌 2. Charger le fichier
    df = pd.read_excel(uploaded_file, sheet_name="All_categ_raw")

    # 📌 3. Vérifier colonne
    col_name = "grounding_search_metadata"
    if col_name not in df.columns:
        st.error(f"La colonne '{col_name}' n'existe pas.")
    else:

        # 📌 4. Fonction pour extraire les URI du JSON
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

        # 📌 5. Extraction
        df["extracted_urls"] = df[col_name].apply(extract_vertex_urls)

        # 📌 6. Exploser (1 URL = 1 ligne)
        df_exploded = df.explode("extracted_urls")

        # 📌 7. Remplacer la colonne d'origine par l’URL seule
        df_exploded[col_name] = df_exploded["extracted_urls"]

        # Supprimer colonne temporaire
        df_exploded = df_exploded.drop(columns=["extracted_urls"])

        # Supprimer lignes sans URL
        df_exploded = df_exploded.dropna(subset=[col_name])

        # 📌 8. Sauvegarder en mémoire
        output = BytesIO()
        df_exploded.to_excel(output, index=False, engine="openpyxl")
        output.seek(0)

        # 📌 9. Télécharger
        st.download_button(
            label="Télécharger le fichier transformé",
            data=output,
            file_name="reformatted_All_categ_raw.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        
