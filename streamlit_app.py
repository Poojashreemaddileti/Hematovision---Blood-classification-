import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HematoVision",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 10%, rgba(55, 135, 255, 0.10), transparent 28%),
        radial-gradient(circle at 10% 80%, rgba(0, 180, 200, 0.07), transparent 25%),
        #f7faff;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Header */
.hero {
    padding: 35px 0 25px 0;
}

.brand {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #155eef;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    line-height: 1.05;
    color: #10233f;
    margin: 18px 0 15px 0;
}

.hero-text {
    color: #60708a;
    font-size: 18px;
    max-width: 680px;
    line-height: 1.7;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.92);
    border: 1px solid #e3eaf4;
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0 15px 45px rgba(24, 55, 95, 0.07);
}

.section-title {
    color: #10233f;
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
}

.section-text {
    color: #718096;
    font-size: 15px;
    line-height: 1.6;
}

/* Result */
.result-card {
    background: linear-gradient(135deg, #ffffff, #f4f8ff);
    border: 1px solid #dce7f5;
    border-radius: 24px;
    padding: 35px;
    margin-top: 25px;
}

.result-label {
    color: #718096;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.result-name {
    color: #155eef;
    font-size: 42px;
    font-weight: 700;
    margin: 8px 0;
    text-transform: capitalize;
}

.confidence {
    color: #10233f;
    font-size: 22px;
    font-weight: 600;
}

/* Info boxes */
.info-box {
    background: #eef5ff;
    border-left: 4px solid #155eef;
    border-radius: 12px;
    padding: 18px;
    color: #40516b;
    line-height: 1.6;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: none;
    padding: 12px 25px;
    font-weight: 600;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: #f8fbff;
    border: 2px dashed #b8cbe5;
    border-radius: 18px;
    padding: 15px;
}

/* Hide Streamlit branding */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")


try:
    model = load_model()
    model_error = None
except Exception as e:
    model = None
    model_error = str(e)


# ---------------- CLASSIFICATION ----------------
CLASS_LABELS = [
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil"
]


def predict_image(uploaded_image):
    img = uploaded_image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(predictions[predicted_index] * 100)

    return predicted_class, confidence


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="brand">HEMATOVISION</div>
    <div class="hero-title">Intelligent Blood<br>Cell Classification</div>
    <div class="hero-text">
        Upload a microscopic blood-cell image and let our
        deep-learning model identify the cell type with
        an AI-powered classification pipeline.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- MODEL ERROR ----------------
if model_error:
    st.error("The trained model could not be loaded.")
    st.code(model_error)
    st.stop()


# ---------------- UPLOAD SECTION ----------------
left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown("""
    <div class="card">
        <div class="section-title">Analyze an Image</div>
        <div class="section-text">
            Upload a blood-cell microscopy image in JPG, JPEG,
            PNG or WEBP format.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload blood-cell image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded microscopy image",
            use_container_width=True
        )

        if st.button("CLASSIFY IMAGE", use_container_width=True):

            with st.spinner("Analyzing blood-cell image..."):
                predicted_class, confidence = predict_image(image)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Classification Result</div>
                <div class="result-name">{predicted_class}</div>
                <div class="confidence">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(min(confidence / 100, 1.0))

            st.markdown("""
            <div class="info-box">
                The result is generated by the trained HematoVision
                deep-learning model. This system is intended for
                educational and research purposes and should not be
                used as a substitute for professional medical diagnosis.
            </div>
            """, unsafe_allow_html=True)


# ---------------- INFORMATION PANEL ----------------
with right:

    st.markdown("""
<div class="card">
<div class="section-title">What HematoVision Detects</div>
<div class="section-text">The model classifies microscopic white blood-cell images into four categories.</div>
<br>
<b>01 — Eosinophil</b><br>
<span class="section-text">White blood cells commonly associated with immune responses.</span>
<br><br>
<b>02 — Lymphocyte</b><br>
<span class="section-text">Important immune-system cells involved in adaptive immunity.</span>
<br><br>
<b>03 — Monocyte</b><br>
<span class="section-text">Large white blood cells involved in immune defense.</span>
<br><br>
<b>04 — Neutrophil</b><br>
<span class="section-text">Abundant white blood cells that play a key role in the body's immune response.</span>
</div>
""", unsafe_allow_html=True)


# ---------------- FOOTER ----------------
st.markdown("""
<br><br>
<div style="
    text-align:center;
    color:#8a98aa;
    font-size:13px;
    padding:25px;
">
    HEMATOVISION · AI-POWERED BLOOD CELL CLASSIFICATION
</div>
""", unsafe_allow_html=True)
