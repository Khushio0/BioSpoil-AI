import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="BioSpoil AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILENAME = "BioSpoil_AI_Best_Model.pkl"
MODEL_PATH = BASE_DIR / MODEL_FILENAME

FEATURES = [
    "Fruit",
    "Temp",
    "Humid (%)",
    "Light (Fux)",
    "CO2 (pmm)"
]

FRUITS = ["Banana", "Orange", "Pineapple", "Tomato"]

GREEN = "#176B48"
DARK_GREEN = "#123B2C"
LIGHT_GREEN = "#E6F1E7"
TEXT = "#263D32"


# =====================================================
# 2. CUSTOM DESIGN AND CSS
# =====================================================

st.markdown("""
<style>

/* Main application background */

.stApp {
    background-color: #F3F7F2;
    color: #263D32;
}

[data-testid="stAppViewContainer"] {
    background-color: #F3F7F2;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Text visibility */

.stApp p,
.stApp label,
.stApp li {
    color: #40574B;
}

.stMarkdown p {
    line-height: 1.7;
    color: #40574B;
}

/* Headings */

h1, h2, h3, h4 {
    color: #173D2D !important;
    font-weight: 750 !important;
}

/* Main title */

.main-title {
    color: #176B48 !important;
    font-size: 43px;
    font-weight: 850;
    letter-spacing: -1px;
}

.subtitle {
    color: #60776A !important;
    font-size: 17px;
    margin-top: 5px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #123B2C;
}

[data-testid="stSidebar"] * {
    color: #F2F8F1 !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D2E5D7 !important;
}

/* Sidebar navigation */

[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 9px;
    padding: 8px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #DCE9DD;
    box-shadow: 0 3px 12px rgba(26,74,48,0.06);
}

[data-testid="stMetricLabel"] {
    color: #60776A !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #176B48 !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    color: #21865A !important;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #176B48;
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 11px 22px;
    font-weight: 650;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #104C33;
    color: white !important;
    border: none;
}

/* Inputs */

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"],
.stTextArea textarea {
    background-color: white !important;
    color: #263D32 !important;
    border-radius: 8px;
}

/* Dataframes */

[data-testid="stDataFrame"] {
    border: 1px solid #DCE9DD;
    border-radius: 12px;
}

/* Alerts */

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Tabs */

.stTabs [data-baseweb="tab"] {
    color: #40574B;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #176B48 !important;
}

/* Dividers */

hr {
    border-color: #DCE8DD;
}

/* Hero section */

.hero-box {
    background: linear-gradient(120deg, #123B2C, #26734D);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero-title {
    color: white !important;
    font-size: 36px;
    font-weight: 850;
}

.hero-description {
    color: #E1F0E5 !important;
    font-size: 16px;
}

/* Food category cards */

.food-card {
    padding: 20px 10px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #E5E9DC;
    min-height: 125px;
}

/* Workflow cards */

.workflow-card {
    background: #FFFFFF;
    border: 1px solid #DCE9DD;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    min-height: 145px;
}

/* Prediction result */

.result-good {
    background: #E5F4E9;
    border: 1px solid #A8D9B5;
    border-radius: 15px;
    padding: 24px;
    text-align: center;
}

.result-bad {
    background: #FCE9E7;
    border: 1px solid #EAB6B1;
    border-radius: 15px;
    padding: 24px;
    text-align: center;
}

/* Footer */

.footer-text {
    text-align: center;
    color: #718578;
    font-size: 13px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = MODEL_PATH

    if not model_path.is_file():

        matching_files = [
            file for file in BASE_DIR.iterdir()
            if file.is_file()
            and file.name.lower() == MODEL_FILENAME.lower()
        ]

        if matching_files:
            model_path = matching_files[0]

    if not model_path.is_file():

        raise FileNotFoundError(
            f"{MODEL_FILENAME} was not found. "
            "Keep the model file in the same directory as app.py."
        )

    return joblib.load(model_path)


try:

    model = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")

    st.info(
        "Ensure BioSpoil_AI_Best_Model.pkl "
        "is uploaded to your GitHub repository."
    )

    st.stop()


# =====================================================
# 4. PREDICTION FUNCTION
# =====================================================

def predict_data(data):

    data = data[FEATURES].copy()

    predictions = model.predict(data)

    probabilities = model.predict_proba(data)

    results = data.copy()

    results["Predicted Class"] = predictions

    results["Model Confidence (%)"] = (
        probabilities.max(axis=1) * 100
    ).round(2)

    for i, label in enumerate(model.classes_):

        results[f"{label} Score (%)"] = (
            probabilities[:, i] * 100
        ).round(2)

    return results


# =====================================================
# 5. RESULT DISPLAY FUNCTION
# =====================================================

def display_prediction(result):

    prediction = str(
        result["Predicted Class"].iloc[0]
    )

    confidence = float(
        result["Model Confidence (%)"].iloc[0]
    )

    st.divider()

    st.subheader("🔬 Analysis Results")

    if prediction.strip().lower() == "good":

        st.markdown(f"""
        <div class="result-good">
            <h2 style="color:#176B48 !important;">
                ✓ GOOD QUALITY
            </h2>
            <p style="color:#176B48 !important;">
                The model predicts the sample belongs
                to the GOOD classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-bad">
            <h2 style="color:#B83F38 !important;">
                ⚠ BAD QUALITY
            </h2>
            <p style="color:#A13F38 !important;">
                The model predicts the sample belongs
                to the BAD classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted Classification",
        prediction.upper()
    )

    col2.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    # Probability chart

    score_columns = [
        f"{label} Score (%)"
        for label in model.classes_
    ]

    probability_data = pd.DataFrame({

        "Class": model.classes_,

        "Probability (%)": [
            result[column].iloc[0]
            for column in score_columns
        ]

    })

    fig = px.bar(
        probability_data,
        x="Class",
        y="Probability (%)",
        color="Class",
        text_auto=".2f",
        color_discrete_map={
            "Good": "#21865A",
            "Bad": "#D9534F"
        },
        title="Model Classification Probability"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#263D32",
        title_font_color="#173D2D",
        yaxis_range=[0, 100],
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "Model confidence represents the model's "
        "classification probability. It is not a direct "
        "measurement of actual food safety, microbial "
        "growth or spoilage."
    )


# =====================================================
# 6. HEADER
# =====================================================

st.markdown(
    '<div class="main-title">🍃 BioSpoil AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Food Quality Classification & Environmental Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown(
    """
    <h1 style="color:white !important;">
        🍃 BioSpoil AI
    </h1>
    <p style="color:#C9E1D1 !important;">
        Food Quality Intelligence
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Single Prediction",
        "Batch Analysis",
        "About the Model"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    <p style="color:#D2E5D7 !important;">
        <b>Powered by</b><br>
        Python<br>
        Scikit-learn<br>
        Streamlit<br>
        Plotly
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.caption(
    "AI Research Prototype | 2026"
)


# =====================================================
# 8. OVERVIEW / HOME PAGE
# =====================================================

if page == "Overview":

    # HERO BANNER

    st.markdown("""
    <div class="hero-box">

        <div class="hero-title">
            🍃 Welcome to BioSpoil AI
        </div>

        <p class="hero-description">
            Smarter Food Quality Classification
            Through Artificial Intelligence
        </p>

        <p style="color:#D0E5D7 !important;">
            Explore how environmental conditions can help
            predict food quality using Machine Learning.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # PLATFORM METRICS

    st.subheader("📊 Platform at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Food Categories",
        "04",
        "Food types"
    )

    c2.metric(
        "Prediction Classes",
        "02",
        "Good / Bad"
    )

    c3.metric(
        "Input Parameters",
        "05",
        "Environmental features"
    )

    c4.metric(
        "ML Algorithm",
        "RF",
        "Random Forest"
    )

    st.divider()

    # FOOD CATEGORIES

    st.subheader("🍎 Explore Food Categories")

    food_columns = st.columns(4)

    food_items = [

        ("🍌", "Banana", "#FFF4D6"),

        ("🍊", "Orange", "#FFE8D6"),

        ("🍍", "Pineapple", "#F3F5D7"),

        ("🍅", "Tomato", "#FFE2DF")

    ]

    for col, item in zip(food_columns, food_items):

        with col:

            st.markdown(f"""
            <div class="food-card"
                 style="background:{item[2]};">

                <div style="font-size:38px;">
                    {item[0]}
                </div>

                <h4 style="color:#173D2D !important;">
                    {item[1]}
                </h4>

            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ENVIRONMENTAL PARAMETERS

    st.subheader("🌍 Environmental Intelligence")

    st.write(
        "Explore the environmental features used by "
        "the machine learning model."
    )

    parameter_info = {

        "Food Type": (
            "Fruit",
            "Categorical",
            "Identifies the food category being analysed."
        ),

        "Temperature": (
            "Temperature (°C)",
            "Numerical",
            "Represents the recorded surrounding temperature."
        ),

        "Humidity": (
            "Humidity (%)",
            "Numerical",
            "Represents the relative humidity of the environment."
        ),

        "Light Intensity": (
            "Light (dataset units)",
            "Numerical",
            "Represents the recorded light intensity."
        ),

        "CO₂": (
            "CO₂ (dataset units)",
            "Numerical",
            "Represents the carbon dioxide measurement "
            "recorded in the dataset."
        )

    }

    selected_parameter = st.selectbox(
        "Select a parameter to explore",
        list(parameter_info.keys())
    )

    details = parameter_info[selected_parameter]

    st.info(
        f"**{details[0]} | {details[1]}**\n\n"
        f"{details[2]}"
    )

    st.divider()

    # HOW IT WORKS

    st.subheader("⚙️ How BioSpoil AI Works")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("""
        <div class="workflow-card">

            <div style="font-size:35px;">📥</div>

            <h3>01. Input</h3>

            <p>
                Enter food type and environmental
                measurements.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with step2:

        st.markdown("""
        <div class="workflow-card">

            <div style="font-size:35px;">🧠</div>

            <h3>02. Analyse</h3>

            <p>
                The trained Random Forest model
                processes the input features.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with step3:

        st.markdown("""
        <div class="workflow-card">

            <div style="font-size:35px;">📊</div>

            <h3>03. Predict</h3>

            <p>
                Obtain a predicted Good/Bad class
                and model confidence.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # INTERACTIVE HOME PAGE PREDICTION

    st.subheader("🔬 Try BioSpoil AI")

    st.write(
        "Enter a sample's environmental conditions "
        "and explore the model prediction directly "
        "from the homepage."
    )

    with st.form("home_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            demo_fruit = st.selectbox(
                "Select Food Category",
                FRUITS,
                key="home_fruit"
            )

            demo_temp = st.number_input(
                "Temperature (°C)",
                value=24.0,
                key="home_temp"
            )

            demo_humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                key="home_humidity"
            )

        with col2:

            demo_light = st.number_input(
                "Light (dataset units)",
                value=12.5,
                key="home_light"
            )

            demo_co2 = st.number_input(
                "CO₂ (dataset units)",
                value=370.0,
                key="home_co2"
            )

        home_submitted = st.form_submit_button(
            "🍃 Analyse Sample",
            use_container_width=True
        )

    if home_submitted:

        demo_data = pd.DataFrame([{

            "Fruit": demo_fruit,

            "Temp": demo_temp,

            "Humid (%)": demo_humidity,

            "Light (Fux)": demo_light,

            "CO2 (pmm)": demo_co2

        }])

        try:

            demo_result = predict_data(demo_data)

            display_prediction(demo_result)

        except Exception as e:

            st.error(f"Prediction error: {e}")

    st.divider()

    st.warning(
        "BioSpoil AI is a research prototype. "
        "Its predictions represent dataset-defined "
        "classification labels and do not establish "
        "actual food safety or microbial spoilage."
    )


# =====================================================
# 9. SINGLE PREDICTION PAGE
# =====================================================

elif page == "Single Prediction":

    st.header("🔬 Individual Food Classification")

    st.write(
        "Enter environmental measurements to classify "
        "an individual food sample."
    )

    st.markdown(
        "**Step 1 — Enter Sample Information**"
    )

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            fruit = st.selectbox(
                "🍎 Select Food Category",
                FRUITS
            )

            temp = st.number_input(
                "🌡️ Temperature (°C)",
                value=24.0
            )

            humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0
            )

        with col2:

            light = st.number_input(
                "☀️ Light (dataset units)",
                value=12.5
            )

            co2 = st.number_input(
                "🌫️ CO₂ (dataset units)",
                value=370.0
            )

        submitted = st.form_submit_button(
            "🔍 Analyse Food Sample",
            use_container_width=True
        )

    if submitted:

        sample = pd.DataFrame([{

            "Fruit": fruit,

            "Temp": temp,

            "Humid (%)": humidity,

            "Light (Fux)": light,

            "CO2 (pmm)": co2

        }])

        try:

            result = predict_data(sample)

            display_prediction(result)

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =====================================================
# 10. BATCH ANALYSIS PAGE
# =====================================================

elif page == "Batch Analysis":

    st.header("📁 Batch Food Classification")

    st.write(
        "Upload a CSV file containing multiple food samples "
        "to perform simultaneous classification."
    )

    st.info(
        "Your CSV must contain all five original model "
        "input features."
    )

    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "📤 Upload your CSV dataset",
        type=["csv"]
    )

    if uploaded_file:

        try:

            batch = pd.read_csv(uploaded_file)

            st.subheader("📋 Uploaded Dataset")

            st.dataframe(
                batch.head(10),
                use_container_width=True
            )

            missing = [

                col for col in FEATURES
                if col not in batch.columns

            ]

            if missing:

                st.error(
                    f"Missing required columns: {missing}"
                )

            elif batch.empty:

                st.error(
                    "The uploaded CSV contains no data."
                )

            elif st.button(
                "🚀 Run Batch Prediction",
                use_container_width=True
            ):

                results = predict_data(batch)

                st.session_state["batch_results"] = results

            if "batch_results" in st.session_state:

                results = st.session_state["batch_results"]

                st.divider()

                st.subheader("📊 Classification Results")

                total_samples = len(results)

                good_count = (
                    results["Predicted Class"]
                    .astype(str)
                    .str.lower()
                    .eq("good")
                    .sum()
                )

                bad_count = total_samples - good_count

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Total Samples",
                    total_samples
                )

                c2.metric(
                    "Good Samples",
                    good_count
                )

                c3.metric(
                    "Bad Samples",
                    bad_count
                )

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.subheader("📈 Classification Distribution")

                counts = (
                    results["Predicted Class"]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "Class",
                    "Number of Samples"
                ]

                fig = px.pie(
                    counts,
                    names="Class",
                    values="Number of Samples",
                    hole=0.45,
                    color="Class",
                    color_discrete_map={
                        "Good": "#21865A",
                        "Bad": "#D9534F"
                    },
                    title="Food Classification Distribution"
                )

                fig.update_layout(
                    paper_bgcolor="white",
                    font_color="#263D32"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                # Download report

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Prediction Report",
                    csv,
                    "BioSpoil_AI_Results.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Batch analysis error: {e}"
            )


# =====================================================
# 11. ABOUT THE MODEL PAGE
# =====================================================

elif page == "About the Model":

    st.header("🧠 About the Machine Learning Model")

    st.write(
        "BioSpoil AI uses a supervised machine learning "
        "approach to classify food samples based on "
        "environmental input parameters."
    )

    st.divider()

    st.subheader("Selected Algorithm")

    st.success("🌳 Random Forest Classifier")

    st.write(
        "Random Forest is an ensemble learning algorithm "
        "that combines multiple decision trees to perform "
        "classification. It uses the collective predictions "
        "of these trees to generate a final output."
    )

    st.divider()

    st.subheader("Input Features")

    parameter_data = pd.DataFrame({

        "Parameter": [
            "Food Category",
            "Temperature",
            "Humidity",
            "Light Intensity",
            "Carbon Dioxide"
        ],

        "Model Feature": FEATURES,

        "Feature Type": [
            "Categorical",
            "Numerical",
            "Numerical",
            "Numerical",
            "Numerical"
        ]

    })

    st.dataframe(
        parameter_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Prediction Classes")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("""
        <div class="result-good">

            <h2 style="color:#176B48 !important;">
                ✓ GOOD
            </h2>

            <p style="color:#176B48 !important;">
                Model-defined Good classification.
            </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown("""
        <div class="result-bad">

            <h2 style="color:#B83F38 !important;">
                ⚠ BAD
            </h2>

            <p style="color:#A13F38 !important;">
                Model-defined Bad classification.
            </p>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("⚙️ Model Architecture")

    st.code("""
    Environmental Input
            |
            v
    Feature Processing
            |
            v
    Random Forest Classifier
            |
            v
    Class Probability Estimation
            |
            v
       GOOD / BAD
    """)

    st.divider()

    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    tech1.metric("Language", "Python")

    tech2.metric("ML Framework", "Scikit-learn")

    tech3.metric("Interface", "Streamlit")

    st.info(
        "Model performance should be interpreted "
        "within the original dataset. External validation "
        "is required before making generalizable claims."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer-text">

    🍃 <b>BioSpoil AI</b><br>

    AI-Based Food Quality Classification & Environmental Analytics<br>

    Machine Learning Research Prototype | 2026

</div>
""", unsafe_allow_html=True)





  
  

 
  

      
