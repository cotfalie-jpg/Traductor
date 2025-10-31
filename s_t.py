import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ==============================
# CONFIGURACIÓN BÁSICA
# ==============================
st.set_page_config(
    page_title="BAE - Analizador TF-IDF 🌼",
    page_icon="🌸",
    layout="centered"
)

# ==============================
# ESTILO VISUAL BAE 🌈
# ==============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fffaf0 0%, #fff8e7 20%, #e3f9f5 60%, #fbe6e4 100%);
        font-family: 'Poppins', sans-serif;
        color: #2b2b2b;
    }

    /* Título principal */
    .main-title {
        font-size: 2.8rem;
        text-align: center;
        background: linear-gradient(90deg, #f9a825, #f48fb1, #81c784, #4db6ac);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1.5s ease-in-out;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #4e4e4e;
        margin-bottom: 2rem;
        animation: fadeIn 2s ease-in-out;
    }

    /* Contenedor de texto */
    .input-box {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(255, 200, 100, 0.2);
        border: 2px solid rgba(255, 230, 150, 0.4);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }

    .input-box:hover {
        transform: scale(1.01);
        box-shadow: 0 6px 25px rgba(255, 200, 150, 0.35);
    }

    /* Botones */
    .stButton button {
        background: linear-gradient(135deg, #ffd54f, #ffb74d);
        color: #2b2b2b;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        width: 100%;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 3px 10px rgba(255, 183, 77, 0.3);
    }

    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 183, 77, 0.5);
    }

    /* Resultados */
    .result-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        border-left: 6px solid #81c784;
        padding: 1.5rem;
        margin-top: 1.5rem;
        animation: fadeInUp 1s ease;
    }

    /* Animaciones suaves */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Tablas */
    .stDataFrame {
        background: #fffef8 !important;
        border-radius: 10px !important;
    }

</style>
""", unsafe_allow_html=True)

# ==============================
# ENCABEZADO
# ==============================
st.markdown('<div class="main-title">🌼 BAE TF-IDF Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Descubre qué texto se parece más a tu pregunta 💭</div>', unsafe_allow_html=True)

# ==============================
# ÁREA DE ENTRADA
# ==============================
with st.container():
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    text_input = st.text_area(
        "✏️ Escribe tus documentos (uno por línea):",
        "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together.",
        height=150
    )
    st.markdown('</div>', unsafe_allow_html=True)

    question = st.text_input("💬 Escribe una pregunta:", "Who is playing?")

# ==============================
# PROCESAMIENTO TF-IDF
# ==============================
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🔍 Analizar Similitudes"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("⚠️ Ingresa al menos un documento.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            stop_words="english",
            token_pattern=None
        )

        X = vectorizer.fit_transform(documents)
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # ==============================
        # RESULTADOS VISUALES
        # ==============================
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### 💡 Pregunta: `{question}`")
        st.markdown(f"**📄 Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.markdown(f"**🎯 Puntaje de similitud:** `{best_score:.3f}`")
        st.markdown('</div>', unsafe_allow_html=True)

        # Tabla TF-IDF
        with st.expander("🔠 Ver Matriz TF-IDF"):
            st.dataframe(df_tfidf.round(3), use_container_width=True)

        # Tabla de similitudes
        with st.expander("📊 Ver Similitudes"):
            sim_df = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": similarities
            })
            st.dataframe(sim_df.sort_values("Similitud", ascending=False), use_container_width=True)

        # Coincidencias de stems
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        if matched:
            st.success(f"✨ Stems coincidentes en el documento elegido: {', '.join(matched)}")
        else:
            st.info("No se encontraron coincidencias exactas de stems.")

# ==============================
# PIE DE PÁGINA
# ==============================
st.markdown("""
---
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
Hecho con 💛 por <strong>BAE</strong> — un toque pastel y un toque de IA 🌸
</div>
""", unsafe_allow_html=True)


