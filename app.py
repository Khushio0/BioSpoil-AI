import html as html_lib
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="BioSpoil AI",
    page_icon=":material/eco:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = "BioSpoil_AI_Best_Model.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME

FEATURES = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)",
]

FRUITS = ["Banana", "Orange", "Pineapple", "Tomato"]

FRUIT_IMAGES = {
    "Banana": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=900&q=85",
    "Orange": "https://images.unsplash.com/photo-1547514701-42782101795e?w=900&q=85",
    "Pineapple": "https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=900&q=85",
    "Tomato": "https://images.unsplash.com/photo-1546094096-0df4bcaaa337?w=900&q=85",
}


# =====================================================
# 2. HTML HELPERS
# =====================================================
# Markdown treats any line indented by 4+ spaces (or any HTML
# followed by a blank line) as a code block. That is what caused
# the raw <h1>, <h3>, <p> tags to appear on the home page.
# render_html() strips indentation and blank lines first, so the
# HTML is always rendered as HTML.

def render_html(markup: str):
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def notice(message: str, kind: str = "info"):
    """Custom message box with guaranteed readable text colours.
    kind: info | warning | success | error"""
    safe = html_lib.escape(str(message))
    render_html(f'<div class="notice notice-{kind}">{safe}</div>')


# =====================================================
# 3. THEME AND CSS
# =====================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F4F7F3;
    color: #20352B;
}

section[data-testid="stMain"] {
    background-color: #F4F7F3;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #174B37 !important;
    font-weight: 800 !important;
}

p, li, label {
    color: #35483E;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #123D30 0%, #176448 100%);
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.35);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent;
    border-radius: 9px;
    padding: 7px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(255,255,255,0.12);
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D6E4D9;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 3px 12px rgba(24,70,48,0.06);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {
    color: #53685B !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #174B37 !important;
    font-weight: 800;
}

/* Buttons (text inside buttons sits in a <p>, so target it too) */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #176B4A !important;
    border: 1px solid #176B4A !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
}

.stButton > button *,
.stFormSubmitButton > button *,
.stDownloadButton > button * {
    color: #FFFFFF !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #104B34 !important;
    border-color: #104B34 !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #20352B !important;
    border-color: #B8CCBE !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] * {
    color: #20352B !important;
}

/* Form */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #D9E5DC;
    border-radius: 15px;
    padding: 22px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #D6E4D9;
    border-radius: 12px;
}

/* Fruit images */
div[data-testid="stImage"] img {
    object-fit: contain !important;
    background-color: #FFFFFF !important;
    border-radius: 13px !important;
    padding: 8px !important;
}

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #D5E2D8 !important;
    border-radius: 15px !important;
    background-color: #FFFFFF;
}

hr {
    border-color: #D5E1D8 !important;
}

a {
    color: #176B4A !important;
}

/* Custom notice boxes (readable in light and dark mode) */
.notice {
    border-radius: 12px;
    border-left: 6px solid;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 15px;
    line-height: 1.65;
    font-weight: 500;
}

.notice-info {
    background: #E3EEF9;
    color: #12345A !important;
    border-color: #2F6DB5;
}

.notice-warning {
    background: #FFF4D6;
    color: #5C4300 !important;
    border-color: #D49A00;
}

.notice-success {
    background: #DFF2E4;
    color: #0F4A2C !important;
    border-color: #247A59;
}

.notice-error {
    background: #FBE3E1;
    color: #7A1F18 !important;
    border-color: #C0392B;
}

/* Sidebar footer */
.sidebar-footer {
    font-size: 13px;
    line-height: 1.7;
}

/* Hero */
.hero {
    background: linear-gradient(115deg, #174B37 0%, #247A59 60%, #70A98A 100%);
    padding: 38px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 7px 22px rgba(20,70,48,0.12);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 36px;
    margin-bottom: 12px;
}

.hero h3 {
    color: #E9F5EC !important;
    font-size: 21px;
    font-weight: 600 !important;
}

.hero p {
    color: #F3FAF4 !important;
    font-size: 16px;
    line-height: 1.8;
}

/* Fruit cards */
.fruit-name {
    color: #174B37;
    font-size: 20px;
    font-weight: 800;
    text-align: center;
}

.fruit-description {
    color: #53685B;
    text-align: center;
    font-size: 14px;
}

/* Workflow */
.workflow-number {
    color: #247A59;
    font-size: 27px;
    font-weight: 800;
}

.workflow-title {
    color: #174B37;
    font-size: 19px;
    font-weight: 800;
}

.workflow-description {
    color: #53685B;
    font-size: 14px;
    line-height: 1.7;
}

/* Parameter explorer */
.param-box {
    background: #E5F0E7;
    border-left: 5px solid #247A59;
    padding: 20px;
    border-radius: 10px;
    margin-top: 10px;
}

.param-box h4 {
    color: #174B37 !important;
}

.param-box p {
    color: #35483E !important;
}

@media (max-width: 768px) {
    .hero { padding: 24px; }
    .hero h1 { font-size: 27px; }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 4. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        matching = [
            f for f in BASE_DIR.iterdir()
            if f.is_file() and f.name.lower() == MODEL_FILENAME.lower()
        ]
        if matching:
            model_path = matching[0]
        else:
            raise FileNotFoundError(
                f"{MODEL_FILENAME} was not found in the application directory."
            )
    else:
        model_path = MODEL_PATH

    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    notice(f"Unable to load trained model: {e}", "error")
    notice(
        "Please check that the trained .pkl file is uploaded "
        "alongside app.py in GitHub.",
        "info",
    )
    st.stop()


# =====================================================
# 5. HEADER
# =====================================================

render_html("""
<div style="margin-bottom:20px;">
<h1 style="color:#174B37;font-size:38px;font-weight:900;margin-bottom:3px;">
BioSpoil AI
</h1>
<p style="color:#53685B;font-size:16px;font-weight:500;">
AI-Based Food Quality Classification &amp; Environmental Analytics
</p>
</div>
""")


# =====================================================
# 6. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("""
<div style="padding:12px 0 20px 0;">
<h2 style="color:white !important;">BioSpoil AI</h2>
<p style="color:#E3F0E7 !important;font-size:14px;">Food Quality Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("### NAVIGATION")

page = st.sidebar.radio(
    "Choose a section",
    ["Overview", "Single Prediction", "Batch Analysis", "About the Model"],
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown("""
<div class="sidebar-footer">
<strong>Technology Stack</strong><br>
Python | Scikit-learn<br>
Pandas | Plotly | Streamlit
<br><br>
<strong>Project Type</strong><br>
Machine Learning Research Prototype
<br><br>
&copy; BioSpoil AI | 2026
</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. PREDICTION FUNCTION
# =====================================================

def predict_data(data):
    data = data[FEATURES].copy()

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    results = data.copy()
    results["Predicted Class"] = predictions
    results["Model Confidence (%)"] = (probabilities.max(axis=1) * 100).round(2)

    for i, label in enumerate(model.classes_):
        results[f"{label} Score (%)"] = (probabilities[:, i] * 100).round(2)

    return results


def show_class_banner(prediction: str, prefix: str):
    value = prediction.strip().lower()
    if value == "good":
        notice(f"{prefix}: GOOD", "success")
    elif value == "bad":
        notice(f"{prefix}: BAD", "error")
    else:
        notice(f"{prefix}: {prediction}", "info")


# =====================================================
# 8. OVERVIEW / HOME PAGE
# =====================================================

if page == "Overview":

    render_html("""
    <div class="hero">
    <h1>Welcome to BioSpoil AI</h1>
    <h3>Smarter Food Quality Classification Through Artificial Intelligence</h3>
    <p>
    Explore how environmental conditions can help predict food quality
    using Machine Learning.
    </p>
    <p>
    An interactive research prototype combining environmental parameters
    and predictive analytics to classify food samples.
    </p>
    </div>
    """)

    st.markdown("## Platform at a Glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food Categories", "04")
    c2.metric("Prediction Classes", "02")
    c3.metric("Input Parameters", "05")
    c4.metric("ML Algorithm", "RF")

    st.divider()

    st.markdown("## Explore Food Categories")
    st.write(
        "Discover the four food categories analysed "
        "by the BioSpoil AI classification model."
    )

    fruit_columns = st.columns(4)

    descriptions = {
        "Banana": "Tropical fruit",
        "Orange": "Citrus fruit",
        "Pineapple": "Tropical fruit",
        "Tomato": "Fresh produce",
    }

    for col, fruit in zip(fruit_columns, FRUITS):
        with col:
            with st.container(border=True):
                st.image(FRUIT_IMAGES[fruit], use_container_width=True)
                render_html(f'<div class="fruit-name">{fruit}</div>')
                render_html(
                    f'<div class="fruit-description">{descriptions[fruit]}</div>'
                )

    st.divider()

    st.markdown("## Environmental Intelligence")
    st.write(
        "Explore the environmental features used by the machine learning model."
    )

    parameter_information = {
        "Food Type": (
            "Fruit",
            "Categorical",
            "Identifies the food category being analysed.",
        ),
        "Temperature": (
            "Temp",
            "Numerical (°C)",
            "Represents the surrounding temperature.",
        ),
        "Humidity": (
            "Humid (%)",
            "Numerical (%)",
            "Represents the relative humidity.",
        ),
        "Light Intensity": (
            "Light (Fux)",
            "Numerical",
            "Represents the light measurement in the dataset.",
        ),
        "CO₂": (
            "CO2 (pmm)",
            "Numerical",
            "Represents the CO2 measurement in the dataset.",
        ),
    }

    selected_parameter = st.selectbox(
        "Select a parameter to explore",
        list(parameter_information.keys()),
    )

    feature, data_type, explanation = parameter_information[selected_parameter]

    render_html(f"""
    <div class="param-box">
    <h4>{feature} | {data_type}</h4>
    <p>{explanation}</p>
    </div>
    """)

    st.divider()

    st.markdown("## How BioSpoil AI Works")

    w1, w2, w3 = st.columns(3)

    workflow = [
        (w1, "01", "Input", "Enter food type and environmental measurements."),
        (w2, "02", "Analyse",
         "The trained Random Forest model processes the input features."),
        (w3, "03", "Predict",
         "Obtain a predicted Good/Bad class and model confidence score."),
    ]

    for col, number, title, text in workflow:
        with col:
            with st.container(border=True):
                render_html(f'<div class="workflow-number">{number}</div>')
                render_html(f'<div class="workflow-title">{title}</div>')
                render_html(f'<div class="workflow-description">{text}</div>')

    st.divider()

    st.markdown("## Try BioSpoil AI")
    st.write(
        "Enter a sample's environmental conditions "
        "and explore the model prediction."
    )

    with st.form("homepage_prediction"):
        col1, col2 = st.columns(2)

        with col1:
            home_fruit = st.selectbox(
                "Select Food Category", FRUITS, key="home_fruit"
            )
            home_temp = st.number_input(
                "Temperature (°C)", value=24.0, key="home_temp"
            )
            home_humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                key="home_humidity",
            )

        with col2:
            home_light = st.number_input(
                "Light (dataset units)", value=12.5, key="home_light"
            )
            home_co2 = st.number_input(
                "CO₂ (dataset units)", value=370.0, key="home_co2"
            )

        home_submit = st.form_submit_button(
            "Analyse Food Quality", use_container_width=True
        )

    if home_submit:
        home_sample = pd.DataFrame([{
            "Fruit": home_fruit,
            "Temp": home_temp,
            "Humid (%)": home_humidity,
            "Light (Fux)": home_light,
            "CO2 (pmm)": home_co2,
        }])

        try:
            home_result = predict_data(home_sample)

            prediction = str(home_result["Predicted Class"].iloc[0])
            confidence = home_result["Model Confidence (%)"].iloc[0]

            st.markdown("### Prediction Result")

            rc1, rc2 = st.columns(2)
            rc1.metric("Predicted Class", prediction)
            rc2.metric("Model Confidence", f"{confidence:.2f}%")

            show_class_banner(prediction, "The model classifies this sample as")

            scores = pd.DataFrame({
                "Class": model.classes_,
                "Score": model.predict_proba(home_sample)[0] * 100,
            })

            fig = px.bar(
                scores,
                x="Class",
                y="Score",
                color="Class",
                text_auto=".2f",
                title="Prediction Probability",
            )
            fig.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#20352B",
                yaxis_range=[0, 100],
            )
            st.plotly_chart(fig, use_container_width=True)

            notice(
                "Model confidence is not a direct measure of food safety "
                "or microbial spoilage.",
                "warning",
            )

        except Exception as e:
            notice(f"Prediction error: {e}", "error")

    st.divider()

    notice(
        "BioSpoil AI is a research prototype. Its predictions represent "
        "dataset-defined classification labels and require external "
        "validation before practical food-safety use.",
        "info",
    )


# =====================================================
# 9. SINGLE PREDICTION
# =====================================================

elif page == "Single Prediction":

    st.markdown("## Individual Food Classification")
    st.write(
        "Enter the environmental conditions to obtain a machine learning prediction."
    )

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            fruit = st.selectbox("Select Food", FRUITS)
            temp = st.number_input("Temperature (°C)", value=24.0)
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0, max_value=100.0, value=90.0
            )

        with col2:
            light = st.number_input("Light (dataset units)", value=12.5)
            co2 = st.number_input("CO₂ (dataset units)", value=370.0)

        submitted = st.form_submit_button(
            "Analyse Food Sample", use_container_width=True
        )

    if submitted:
        sample = pd.DataFrame([{
            "Fruit": fruit,
            "Temp": temp,
            "Humid (%)": humidity,
            "Light (Fux)": light,
            "CO2 (pmm)": co2,
        }])

        try:
            result = predict_data(sample)

            prediction = str(result["Predicted Class"].iloc[0])
            confidence = result["Model Confidence (%)"].iloc[0]

            st.divider()
            st.markdown("### Prediction Results")

            show_class_banner(prediction, "Predicted Classification")

            c1, c2 = st.columns(2)
            c1.metric("Predicted Class", prediction)
            c2.metric("Model Confidence", f"{confidence:.2f}%")

            score_columns = [f"{label} Score (%)" for label in model.classes_]

            probability_data = pd.DataFrame({
                "Class": model.classes_,
                "Model Score (%)": [
                    result[column].iloc[0] for column in score_columns
                ],
            })

            fig = px.bar(
                probability_data,
                x="Class",
                y="Model Score (%)",
                color="Class",
                title="Predicted Class Scores",
                text_auto=".2f",
            )
            fig.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#20352B",
            )
            st.plotly_chart(fig, use_container_width=True)

            notice(
                "Model confidence is not a measure of actual food safety "
                "or microbial spoilage.",
                "warning",
            )

        except Exception as e:
            notice(f"Prediction error: {e}", "error")


# =====================================================
# 10. BATCH ANALYSIS
# =====================================================

elif page == "Batch Analysis":

    st.markdown("## Batch Food Classification")
    st.write(
        "Upload a CSV file containing the five required model input features."
    )

    st.markdown("### Required CSV Columns")
    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file:
        try:
            batch = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Dataset")
            st.dataframe(batch.head(), use_container_width=True)

            missing = [col for col in FEATURES if col not in batch.columns]

            if missing:
                notice(f"Missing required columns: {missing}", "error")

            elif batch.empty:
                notice("The uploaded CSV is empty.", "warning")

            elif st.button("Run Batch Prediction", use_container_width=True):

                results = predict_data(batch)

                st.markdown("### Classification Results")
                st.dataframe(results, use_container_width=True)

                st.markdown("### Class Distribution")

                counts = results["Predicted Class"].value_counts().reset_index()
                counts.columns = ["Class", "Number of Samples"]

                fig = px.pie(
                    counts,
                    names="Class",
                    values="Number of Samples",
                    title="Predicted Food Classification",
                    hole=0.4,
                )
                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    font_color="#20352B",
                )
                st.plotly_chart(fig, use_container_width=True)

                st.metric("Total Samples Analysed", len(results))

                csv = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Prediction Report",
                    csv,
                    "BioSpoil_AI_Results.csv",
                    "text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            notice(f"Batch analysis error: {e}", "error")


# =====================================================
# 11. ABOUT THE MODEL
# =====================================================

elif page == "About the Model":

    st.markdown("## Machine Learning Model")

    st.markdown("### Selected Algorithm")
    notice("Random Forest Classifier", "success")

    st.write(
        """
        Random Forest is an ensemble learning algorithm that combines
        multiple decision trees to perform classification.

        It generates predictions by combining the outputs of multiple
        decision trees.
        """
    )

    st.divider()

    st.markdown("### Model Input Features")

    parameter_data = pd.DataFrame({
        "Parameter": ["Fruit", "Temperature", "Humidity", "Light", "CO₂"],
        "Feature": FEATURES,
        "Type": [
            "Categorical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical",
        ],
    })

    st.dataframe(parameter_data, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Prediction Classes")

    c1, c2 = st.columns(2)

    with c1:
        with st.container(border=True):
            st.markdown("### GOOD")
            st.write(
                "The sample is classified as Good according to the trained model."
            )

    with c2:
        with st.container(border=True):
            st.markdown("### BAD")
            st.write(
                "The sample is classified as Bad according to the trained model."
            )

    st.divider()

    st.markdown("### Model Architecture")

    st.code("""
Environmental Input
        |
Feature Processing
        |
Random Forest Classifier
        |
Class Probability Estimation
        |
Good / Bad Prediction
    """)

    notice(
        "The reported model performance should be interpreted within the "
        "original dataset. External validation is necessary before making "
        "generalizable claims.",
        "info",
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

render_html("""
<div style="text-align:center;padding:18px;background:#E6EFE7;border-radius:12px;">
<h3 style="color:#174B37 !important;">BioSpoil AI</h3>
<p style="color:#35483E;">
AI-Based Food Quality Classification &amp; Environmental Analytics
</p>
<p style="color:#53685B;font-size:13px;">
Machine Learning Research Prototype | 2026
</p>
</div>
""")
  
