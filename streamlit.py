import os
import time
import json
import html
import ast
import pandas as pd
import streamlit as st

import main as backend

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="RetailNext — AI powered Stylist",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Theming (AltaDaily-inspired) -------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');

      :root{
        --bg: #ffffff;         /* white background to match images */
        --text: #1b1b1b;       /* charcoal */
        --muted: #6e6e6e;
        --accent: #d1654b;     /* burnt coral */
        --card: #ffffff;
        --border: rgba(0,0,0,0.08);
      }

      html, body, .stApp {
        background: var(--bg);
        color: var(--text);
        font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }

      .block-container {
        pading-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1100px;
      }

      h1, h2, h3, .hero-title {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif;
        letter-spacing: 0.2px;
      }

      /* Top header: black background with white text/icons */
      header, [data-testid="stHeader"] {
        background: #000 !important;
        color: #fff !important;
      }
      header *, [data-testid="stHeader"] * {
        color: #fff !important;
        fill: #fff !important;
      }

      /* Hero */
      .hero {
        background: linear-gradient(180deg, rgba(0,0,0,0.03), rgba(0,0,0,0.00));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px 28px 22px 28px;
        margin-bottom: 18px;
      }
      .brand {
        display: inline-flex;
        gap: 10px;
        align-items: center;
        color: var(--text);
        text-decoration: none;
      }
      .brand-badge {
        background: var(--text);
        color: var(--bg);
        font-weight: 700;
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 13px;
        letter-spacing: .6px;
        text-transform: uppercase;
      }
      .brand-logo {
        display: inline-flex;
        align-items: center;
      }
      .brand-logo svg {
        height: 44px;
      }
      .hero-title {
        margin: 10px 0 6px 0;
        font-size: 40px;
        line-height: 1.12;
      }
      .hero-sub {
        color: #fff;
        font-size: 16px;
        margin-bottom: 6px;
      }

      /* Hero */
      .hero {
        background: #000;
        color: #fff;
        border: 1px solid #000;
        border-radius: 16px;
        padding: 28px 28px 22px 28px;
        margin-bottom: 18px;
      }
      .hero .accent {
        color: #fff !important;
      }

      /* Cards */
      .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
      }

      /* Accents */
      .accent {
        color: var(--accent);
      }

      /* File uploader + buttons */
      .stFileUploader label {
        font-weight: 600;
        font-size: 0.95rem;
      }
      button[kind="primary"], .stButton>button {
        background: var(--accent);
        color: #fff;
        border: 1px solid var(--accent);
        border-radius: 10px;
        font-weight: 600;
      }
      .stButton>button:hover {
        filter: brightness(0.96);
      }

      /* JSON/code blocks */
      pre, code {
        font-size: 0.88rem !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Helpers ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_embeddings():
    # Load catalog and precomputed embeddings
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    _ = pd.read_csv(styles_filepath, on_bad_lines='skip')  # ensure base file exists

    styles_df = pd.read_csv('data/sample_clothes/sample_styles_with_embeddings.csv', on_bad_lines='skip')
    styles_df['embeddings'] = styles_df['embeddings'].apply(lambda x: ast.literal_eval(x))
    return styles_df

def save_uploaded_file(uploaded_file) -> str:
    uploads_dir = "data/uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    filename = f"reference_{int(time.time())}{ext or '.jpg'}"
    path = os.path.join(uploads_dir, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# ---- Layout -----------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="brand">
        <div class="brand-logo" aria-label="RetailNext logo">
          <svg viewBox="0 0 360 60" xmlns="http://www.w3.org/2000/svg" role="img">
            <title>RetailNext</title>
            <!-- Icon: new RN monogram emblem (optimized for black background) -->
            <defs>
              <linearGradient id="rnAccent" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stop-color="#ffffff" stop-opacity="1"/>
                <stop offset="100%" stop-color="#d1654b" stop-opacity="1"/>
              </linearGradient>
            </defs>
            <g aria-label="RetailNext monogram emblem">
              <!-- Circular outline -->
              <circle cx="28" cy="30" r="22" fill="none" stroke="#ffffff" stroke-width="2"/>
              <!-- Stylized R -->
              <path d="M18 40 V20 h10 a6 6 0 1 1 0 12 h-10 l14 8" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <!-- Accent N strokes -->
              <path d="M34 21 L34 39 M34 21 L42 39 M42 21 L42 39" fill="none" stroke="#d1654b" stroke-width="2" stroke-linecap="round"/>
              <!-- Sparkle detail -->
              <g stroke="#ffffff" stroke-width="1.5" stroke-linecap="round">
                <path d="M46 12 v6"/>
                <path d="M43 15 h6"/>
              </g>
            </g>
            <!-- Wordmark (larger, white) -->
            <g transform="translate(60, 0)">
              <text x="0" y="38" font-family="'Playfair Display', Georgia, serif" font-size="34" font-weight="700">
                <tspan fill="#ffffff">Retail</tspan><tspan fill="#ffffff">Next</tspan>
              </text>
            </g>
          </svg>
        </div>
      </div>
      <div class="hero-title">AI powered Stylist</div>
      <div class="hero-sub">Upload a clothing item image and let AI style the rest of the outfit — <span class="accent">curated</span> to match your vibe.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a clothing item image", type=["jpg", "jpeg", "png"])
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown(
        """
        <div class="card">
          <strong>How it works</strong>
          <ul style="margin-top:.5rem;">
            <li>We analyze your uploaded item to infer category and style.</li>
            <li>We search a catalog for items that pair well with it.</li>
            <li>Optionally, we verify AI pairing compatibility.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

styles_df = get_embeddings()

if uploaded:
    # Save and display the uploaded image
    reference_image_path = save_uploaded_file(uploaded)
    st.markdown("#### Your item")
    st.image(reference_image_path, caption="Uploaded item", use_container_width=False)

    # Prepare subcategories
    unique_subcategories = styles_df['articleType'].unique()

    # Encode uploaded image and run analysis
    with st.spinner("Analyzing your item..."):
        encoded_image = backend.encode_image_to_base64(reference_image_path)
        analysis = backend.analyze_image(encoded_image, unique_subcategories)

    # Parse analysis safely
    image_analysis = {}
    try:
        image_analysis = json.loads(analysis)
    except Exception:
        # Fall back to best effort if response isn't clean JSON
        try:
            image_analysis = json.loads(analysis.strip().strip("`"))
        except Exception:
            st.error("Couldn't parse analysis response. Please try another image.")
            st.code(analysis)
            st.stop()

    # Extract features
    item_descs = image_analysis.get('items', [])
    item_category = image_analysis.get('category', '')
    item_gender = image_analysis.get('gender', '')

    st.markdown("#### Styling suggestions")
    # Build pretty analysis card (non-JSON)
    item_pills = ""
    if item_descs:
        for it in item_descs:
            item_pills += f'<span style="border:1px solid rgba(0,0,0,0.08);background:#fff;border-radius:999px;padding:6px 10px;font-size:13px; display:inline-block; margin:2px 4px 2px 0;">{html.escape(str(it))}</span>'
    analysis_html = f"""
    <div class="card">
      <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
        <span style="background:#1b1b1b;color:#fff;border-radius:8px;padding:6px 10px;font-weight:600;font-size:13px;letter-spacing:.4px;">Category: {html.escape(str(item_category)) if item_category else '—'}</span>
        <span style="background:#d1654b;color:#fff;border-radius:8px;padding:6px 10px;font-weight:600;font-size:13px;letter-spacing:.4px;">Gender: {html.escape(str(item_gender)) if item_gender else '—'}</span>
      </div>
      <div style="margin-top:2px;">
        <div style="font-weight:600;margin-bottom:6px;">Suggested pairing items</div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">{item_pills or '<em style="color:#6e6e6e;">No items suggested</em>'}</div>
      </div>
    </div>
    """
    st.markdown(analysis_html, unsafe_allow_html=True)

    # Guard against empty analysis
    if not item_descs or not item_category or not item_gender:
        st.warning("Incomplete analysis received. Try a clearer image.")
        st.stop()

    # Filter catalog by gender/unisex and different category
    filtered_items = styles_df.loc[styles_df['gender'].isin([item_gender, 'Unisex'])]
    filtered_items = filtered_items[filtered_items['articleType'] != item_category]
    st.caption(f"{len(filtered_items)} candidate items in catalog after filtering")

    # Match against item descriptions
    with st.spinner("Finding curated matches..."):
        matching_items = backend.find_matching_items_with_rag(filtered_items, item_descs)

    # Generate candidate match image paths (hidden from display)
    paths = []
    for i, item in enumerate(matching_items):
        item_id = item['id']
        image_path = f'data/sample_clothes/sample_images/{item_id}.jpg'
        paths.append(image_path)

    # Optional compatibility check for unique paths
    unique_paths = list(set(paths))
    if unique_paths:
        st.markdown("#### Outfit matching")
        with st.spinner("Evaluating outfit compatibility..."):
            for path in unique_paths:
                try:
                    suggested_image = backend.encode_image_to_base64(path)
                    raw = backend.check_match(encoded_image, suggested_image)
                    if raw:
                        verdict = json.loads(raw)
                        if verdict.get("answer") == "yes":
                            st.image(path, caption="Matched pairing", use_container_width=False)
                            st.success(verdict.get("reason", "Looks good together!"))
                except Exception:
                    continue
else:
    st.info("Upload a clothing item image to get started.")
