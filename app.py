import streamlit as st
import pandas as pd
import numpy as np
import torch
import open_clip
import faiss
import os
import base64

# ---------- Page config ----------
st.set_page_config(page_title="Muse", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&display=swap');

/* Overall background — off-white parchment */
.stApp {
    background-color: #F4EFE4;
}

#MainMenu, footer, header {visibility: hidden;}

div[data-testid="stTextInput"] {
    max-width: 800px;
    margin: 1.5rem auto 2.5rem auto;
}

/* Search bar — white, bigger, icon baked in as a background image (avoids
   nesting issues with separate floating elements) */
.stApp input {
    background-color: #FFFFFF !important;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='22' height='22' viewBox='0 0 24 24' fill='none'><circle cx='11' cy='11' r='7' stroke='%233A3939' stroke-width='2'/><line x1='16.5' y1='16.5' x2='21' y2='21' stroke='%233A3939' stroke-width='2' stroke-linecap='round'/></svg>") !important;
    background-repeat: no-repeat !important;
    background-position: 24px center !important;
    padding: 22px 50px 22px 60px !important;
    font-size: 19px !important;
    color: #3A3939 !important;
    box-shadow: 0 3px 12px rgba(0,0,0,0.07) !important;
}

.stApp input::placeholder {
    color: #4A4948 !important;
    font-weight: 500 !important;
}

.stApp input:focus {
    box-shadow: 0 3px 16px rgba(0,0,0,0.13) !important;
}

/* Title — Muse, in Playfair Display, thinner, smaller */
.app-title {
    text-align: center;
    color: #0A0A0A;
    font-family: 'Playfair Display', serif;
    font-weight: 200;
    font-size: 175px;
    padding-top: 1.5rem;
    margin-bottom: 0.5rem;
    letter-spacing: 0.5px;
}

/* Pinterest-style masonry grid using CSS columns */
.masonry {
    column-count: 4;
    column-gap: 16px;
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 1rem;
}
.masonry-item {
    break-inside: avoid;
    margin-bottom: 16px;
    background: white;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    transition: transform 0.15s ease;
}
.masonry-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}

/* Image crop/zoom wrapper — crops excess white margin from source photos
   and zooms in on the product so it fills more of the card */
.masonry-item .img-crop {
    width: 100%;
    height: 260px;
    overflow: hidden;
    position: relative;
}
.masonry-item .img-crop img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transform: scale(0.5);
    transform-origin: center;
    border-radius: 0;
}

.masonry-caption {
    padding: 10px 12px 12px 12px;
    font-family: 'Helvetica Neue', sans-serif;
    font-size: 13px;
    color: #4A4948;
}

@media (max-width: 1000px) {
    .masonry { column-count: 3; }
}
@media (max-width: 700px) {
    .masonry { column-count: 2; }
}
</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="app-title">Muse</div>', unsafe_allow_html=True)

# ---------- Search bar ----------
query = st.text_input(" ", placeholder="Search", label_visibility="collapsed")

# ---------- Load model + index (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device

@st.cache_resource
def load_index():
    embeddings = np.load("data/image_embeddings.npy")
    ids_df = pd.read_csv("data/embedded_ids.csv")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, ids_df

model, tokenizer, device = load_model()
index, ids_df = load_index()

def search(query_text, k=16):
    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        text_embedding = model.encode_text(tokens)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(text_embedding, k)
    results = ids_df.iloc[indices[0]].copy()
    results["score"] = scores[0]
    return results

# ---------- Results grid ----------
if query:
    results = search(query, k=16)

    html_parts = ['<div class="masonry">']
    for _, row in results.iterrows():
        img_path = f"data/images/{row['id']}.jpg"
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            item_html = (
                '<div class="masonry-item">'
                '<div class="img-crop">'
                f'<img src="data:image/jpeg;base64,{b64}" />'
                '</div>'
                f'<div class="masonry-caption">{row["productDisplayName"]}</div>'
                '</div>'
            )
            html_parts.append(item_html)
    html_parts.append('</div>')

    st.markdown(''.join(html_parts), unsafe_allow_html=True)