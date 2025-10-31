import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# ======================
# CONFIGURACIÓN DE PÁGINA
# ======================
st.set_page_config(
    page_title="TF-IDF BAE 🌸",
    page_icon="🍼",
    layout="centered"
)

# ======================
# ESTILO BAE 🌼 (colores pastel + animaciones suaves)
# ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

.stApp {
    background: linear-gradient(180deg, #FFF8E1 0%, #FFF2C3 50%, #C6E2E3 100%);
    color: #3E3E3E;
    font-family: 'Poppins', sans-serif;
}

/* Título */
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    color: #355C4A;
    margin-bottom: 0.3rem;
    animation: fadeInDown 1.2s ease;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #5F7161;
    margin-bottom: 1.8rem;
    animation: fadeIn 1.5s ease;
}

/* Cajas */
.bae-box {
    background: #FFF8EA;
    border-radius: 20px;
    border: 2px solid #DD8E6B;
    padding: 1.5rem;
    box-shadow: 0 5px 20px rgba(221, 142, 107, 0.15);
    animation: fadeIn 1.2s ease;
}

/* Botones */
.stButton>button {
    background: linear-gradient(135deg, #DD8E6B, #FFC98B);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 0.8rem 2rem;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(221, 142, 107, 0.3);
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(198, 226, 227, 0.6);
}

/* Tablas */
.dataframe {
    border-radius: 12px !important;
}

/* Animaciones */
@keyframes fadeIn { from {opacity: 0;} to {opacity: 1;} }
@keyframes fadeInDown { from {opacity: 0; transform: translateY(-20px);} to {opacity: 1; transform: translateY(0);} }

/* Stems */
.matched-stem {
    display: inline-block;
    background: #C6E2E3;
    color: #355C4A;
    border-radius: 10px;
    padding: 6px 10px;
    margin: 5px;
    font-size: 0.9rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
.matched-stem:hover {
    background: #FFF2C3;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ======================
# INTERFAZ
# ======================
st.markdown('<div class="main-title">🌿 TF-IDF BAE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Compara textos con un toque cálido y pastel 🍼</div>', unsafe_allow_html=True)

st.markdown('<div class="bae-box">', unsafe_allow_html=True)
st.write("""
Cada línea se trata como un **documento** (una frase o párrafo).  
💡 Los textos deben estar en **inglés**, y las palabras similares (*play*, *playing*) se agrupan con *stemming*.
""")

# Ejemplo inicial
text_input = st.text_area(
    "✏️ Escribe tus documentos (uno por línea, en inglés):",
    "The dog barks loudly.\nThe cat meows at night.\nThe dog and the cat play together."
)

question = st.text_input("💬 Escribe una pregunta (en inglés):", "Who is playing?")

# Stemmer inglés
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text: str):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

if st.button("🌸 Calcular TF-IDF y buscar respuesta"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    if len(documents) < 1:
        st.warning("Por favor, ingresa al menos un documento 🌼")
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

        st.markdown("### 📊 Matriz TF-IDF (stems)")
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # Pregunta y similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        st.markdown("### 💡 Resultado principal")
        st.success(f"**Documento más relevante (Doc {best_idx+1}):** {best_doc}")
        st.markdown(f"**Puntaje de similitud:** `{best_score:.3f}`")

        # Mostrar todas las similitudes
        st.markdown("### 🌈 Puntajes de similitud")
        sim_df = pd.DataFrame({
            "Documento": [f"Doc {i+1}" for i in range(len(documents))],
            "Texto": documents,
            "Similitud": similarities
        })
        st.dataframe(sim_df.sort_values("Similitud", ascending=False), use_container_width=True)

        # Stems coincidentes
        vocab = vectorizer.get_feature_names_out()
        q_stems = tokenize_and_stem(question)
        matched = [s for s in q_stems if s in vocab and df_tfidf.iloc[best_idx].get(s, 0) > 0]

        st.markdown("### ✨ Stems de la pregunta encontrados:")
        if matched:
            for s in matched:
                st.markdown(f"<span class='matched-stem'>{s}</span>", unsafe_allow_html=True)
        else:
            st.info("No se encontraron coincidencias exactas 🌻")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; margin-top:2rem; color:#5F7161; font-size:0.9rem;'>
Hecho con 💛 por <b>BAE</b> | Similitudes que iluminan 🌿
</div>
""", unsafe_allow_html=True)


        
    


