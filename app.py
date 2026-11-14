import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
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


# =====================================================
# 2. PROFESSIONAL THEME
# =====================================================

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #F3F7F2;
    color: #263D32;
}

[data-testid="stMain"] {
    background-color: #F3F7F2;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
.stApp h1, .stApp h2, .stApp h3,
.stApp h4 {
    color: #174C36 !important;
    font-weight: 750 !important;
}

/* Normal text */
.stApp p, .stApp li,
.stApp label {
    color: #40574B !important;
}

[data-testid="stCaptionContainer"] {
    color: #536B5C !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #123B2C;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #D7E5D9;
    border-radius: 15px;
    padding: 19px;
    box-shadow: 0 3px 12px rgba(20,70,45,0.05);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #536B5C !important;
}

[data-testid="stMetricValue"] {
    color: #176B48 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button,
.stDownloadButton > button {
    background-color: #176B48 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 43px;
}

.stButton > button p,
.stFormSubmitButton > button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}

.stButton > button:hover {
    background-color: #104C33 !important;
    border: 1px solid #104C33 !important;
}

/* Input boxes */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background-color: #FFFFFF !important;
    color: #263D32 !important;
    -webkit-text-fill-color: #263D32 !important;
    border: 1px solid #B9CDBD !important;
    border-radius: 9px !important;
}

/* Select boxes */
.stSelectbox [data-baseweb="select"] {
    background-color: #FFFFFF !important;
}

.stSelectbox [data-baseweb="select"] * {
    color: #263D32 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] * {
    color: #263D32 !important;
}

/* Forms */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE9DD;
    border-radius: 15px;
    padding: 20px;
}

/* Fruit photographs */
[data-testid="stImage"] img {
    height: 190px !important;
    width: 100% !important;
    object-fit: cover !important;
    border-radius: 12px !important;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #D8E5D9 !important;
    border-radius: 15px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #9CB9A2;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] * {
    color: #263D32 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #536B5C;
    padding: 18px;
    font-size: 13px;
}

hr {
    border-color: #D8E5D9 !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    model_path = MODEL_PATH

    if not model_path.is_file():

        matching_files = [
            f for f in BASE_DIR.iterdir()
            if f.is_file()
            and f.name.lower() == MODEL_FILENAME.lower()
        ]

        if matching_files:
            model_path = matching_files[0]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"{MODEL_FILENAME} not found. "
            "Keep the model in the same folder as app.py."
        )

    return joblib.load(model_path)


try:
    model = load_model()

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.info("Check your GitHub model file.")
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
# 5. DISPLAY PREDICTION RESULTS
# =====================================================

def display_prediction(result):

    prediction = str(
        result["Predicted Class"].iloc[0]
    )

    confidence = float(
        result["Model Confidence (%)"].iloc[0]
    )

    st.divider()
    st.subheader("🔬 Prediction Results")

    if prediction.strip().lower() == "good":

        st.success(
            "✓ Predicted Classification: GOOD"
        )

    else:

        st.error(
            "⚠ Predicted Classification: BAD"
        )

    c1, c2 = st.columns(2)

    c1.metric(
        "Predicted Class",
        prediction.upper()
    )

    c2.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    probability_data = pd.DataFrame({

        "Class": list(model.classes_),

        "Probability (%)": [
            result[f"{label} Score (%)"].iloc[0]
            for label in model.classes_
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
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font_color="#263D32",
        yaxis_range=[0, 100],
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "Model confidence is not a direct measurement "
        "of actual food safety or microbial spoilage."
    )


# =====================================================
# 6. HEADER
# =====================================================

st.title("🍃 BioSpoil AI")

st.caption(
    "AI-Based Food Quality Classification & Environmental Analytics"
)


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🍃 BioSpoil AI")

st.sidebar.caption("Food Quality Intelligence")

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

st.sidebar.caption(
    "Developed using Python, Scikit-learn and Streamlit"
)

st.sidebar.caption("Research Prototype | 2026")


# =====================================================
# 8. OVERVIEW / HOMEPAGE
# =====================================================

if page == "Overview":

    # HERO SECTION - NATIVE STREAMLIT ONLY

    with st.container(border=True):

        st.markdown("## 🍃 Welcome to BioSpoil AI")

        st.subheader(
            "Smarter Food Quality Classification "
            "Through Artificial Intelligence"
        )

        st.write(
            "Explore how environmental conditions can help "
            "predict food quality using Machine Learning."
        )

        st.write(
            "An interactive research prototype combining "
            "environmental parameters and predictive analytics."
        )

        if st.button(
            "Explore Food Analysis ↓",
            use_container_width=False
        ):
            st.session_state["explore_food"] = True

    if st.session_state.get("explore_food", False):
        st.info(
            "Explore the food categories below or navigate "
            "to Single Prediction to analyse a sample."
        )

    st.write("")

    # PLATFORM METRICS

    st.subheader("📊 Platform at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Food Categories", "04")
    c2.metric("Prediction Classes", "02")
    c3.metric("Input Parameters", "05")
    c4.metric("ML Algorithm", "RF")

    st.divider()

    # FOUR FRUIT IMAGE CARDS

    st.subheader("🍎 Explore Food Categories")

    st.write(
        "Discover the four food categories analysed "
        "by the BioSpoil AI classification model."
    )

    fruits = [

        {
            "name": "Banana",
            "emoji": "🍌",
            "description": "Tropical fruit",
            "image": (
                "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e"
                "?auto=format&fit=crop&w=700&q=85"
            )
        },

        {
            "name": "Orange",
            "emoji": "🍊",
            "description": "Citrus fruit",
            "image": (
                "https://images.unsplash.com/photo-1547514701-42782101795e"
                "?auto=format&fit=crop&w=700&q=85"
            )
        },

        {
            "name": "Pineapple",
            "emoji": "🍍",
            "description": "Tropical fruit",
            "image": (
                "https://images.unsplash.com/photo-1550258987-190a2d41a8ba"
                "?auto=format&fit=crop&w=700&q=85"
            )
        },

        {
            "name": "Tomato",
            "emoji": "🍅",
            "description": "Fresh produce",
            "image": (
                "https://images.unsplash.com/photo-1546094096-0df4bcaaa337"
                "?auto=format&fit=crop&w=700&q=85"
            )
        }

    ]

    fruit_columns = st.columns(4, gap="medium")

    for col, fruit in zip(fruit_columns, fruits):

        with col:

            with st.container(border=True):

                st.image(
                    fruit["image"],
                    use_container_width=True
                )

                st.markdown(
                    f"### {fruit['emoji']} {fruit['name']}"
                )

                st.caption(fruit["description"])

    st.divider()

    # ENVIRONMENTAL INTELLIGENCE

    st.subheader("🌍 Environmental Intelligence")

    st.write(
        "Explore the environmental features used "
        "by the machine learning model."
    )

    parameter_info = {

        "Food Type": (
            "Fruit | Categorical",
            "Identifies the food category being analysed."
        ),

        "Temperature": (
            "Temperature (°C) | Numerical",
            "Represents the recorded surrounding temperature."
        ),

        "Humidity": (
            "Humidity (%) | Numerical",
            "Represents the relative humidity of the environment."
        ),

        "Light Intensity": (
            "Light (dataset units) | Numerical",
            "Represents the recorded light intensity."
        ),

        "CO₂": (
            "CO₂ (dataset units) | Numerical",
            "Represents the carbon dioxide measurement "
            "recorded in the dataset."
        )

    }

    selected_parameter = st.selectbox(
        "Select a parameter to explore",
        list(parameter_info.keys())
    )

    with st.container(border=True):

        st.markdown(
            f"### {parameter_info[selected_parameter][0]}"
        )

        st.write(
            parameter_info[selected_parameter][1]
        )

    st.divider()

    # HOW IT WORKS

    st.subheader("⚙️ How BioSpoil AI Works")

    step1, step2, step3 = st.columns(3)

    with step1:

        with st.container(border=True):

            st.markdown("## 📥")

            st.markdown("### 01. Input")

            st.write(
                "Enter food type and environmental measurements."
            )

    with step2:

        with st.container(border=True):

            st.markdown("## 🧠")

            st.markdown("### 02. Analyse")

            st.write(
                "The trained Random Forest model processes "
                "the input features."
            )

    with step3:

        with st.container(border=True):

            st.markdown("## 📊")

            st.markdown("### 03. Predict")

            st.write(
                "Obtain a Good/Bad classification "
                "and model confidence."
            )

    st.divider()

    # INTERACTIVE HOMEPAGE PREDICTION

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
                "🍎 Select Food Category",
                FRUITS,
                key="home_fruit"
            )

            demo_temp = st.number_input(
                "🌡️ Temperature (°C)",
                value=24.0,
                key="home_temp"
            )

            demo_humidity = st.number_input(
                "💧 Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                key="home_humidity"
            )

        with col2:

            demo_light = st.number_input(
                "☀️ Light (dataset units)",
                value=12.5,
                key="home_light"
            )

            demo_co2 = st.number_input(
                "🌫️ CO₂ (dataset units)",
                value=370.0,
                key="home_co2"
            )

        home_submitted = st.form_submit_button(
            "🍃 Analyse Food Sample",
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

    st.info(
        "BioSpoil AI is a research prototype. "
        "Predictions represent dataset-defined classification "
        "labels and do not establish actual food safety "
        "or microbial spoilage."
    )


# =====================================================
# 9. SINGLE PREDICTION
# =====================================================

elif page == "Single Prediction":

    st.header("🔬 Individual Food Classification")

    st.write(
        "Enter environmental conditions to obtain "
        "an AI-based classification of your food sample."
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
# 10. BATCH ANALYSIS
# =====================================================

elif page == "Batch Analysis":

    st.header("📁 Batch Food Classification")

    st.write(
        "Upload a CSV file containing multiple food samples "
        "to perform simultaneous classification."
    )

    st.info(
        "Your CSV must contain all five original "
        "model input features."
    )

    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "📤 Upload CSV Dataset",
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

                st.session_state["batch_results"] = (
                    predict_data(batch)
                )

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

                c1.metric("Total Samples", total_samples)
                c2.metric("Good Samples", good_count)
                c3.metric("Bad Samples", bad_count)

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
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font_color="#263D32"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

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

            st.error(f"Batch analysis error: {e}")


# =====================================================
# 11. ABOUT THE MODEL
# =====================================================

elif page == "About the Model":

    st.header("🧠 About the Machine Learning Model")

    st.write(
        "BioSpoil AI uses supervised machine learning "
        "to classify food samples using environmental "
        "input parameters."
    )

    st.divider()

    st.subheader("Selected Algorithm")

    st.success("🌳 Random Forest Classifier")

    st.write(
        "Random Forest is an ensemble learning algorithm "
        "that combines multiple decision trees to perform "
        "classification."
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

        with st.container(border=True):

            st.success("✓ GOOD")

            st.write(
                "Model-defined Good classification."
            )

    with c2:

        with st.container(border=True):

            st.error("⚠ BAD")

            st.write(
                "Model-defined Bad classification."
            )

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

    c1, c2, c3 = st.columns(3)

    c1.metric("Programming", "Python")
    c2.metric("ML Framework", "Scikit-learn")
    c3.metric("Interface", "Streamlit")

    st.info(
        "Model performance should be interpreted "
        "within the original dataset. External validation "
        "is needed before making generalizable claims."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        🍃 <b>BioSpoil AI</b>
        <br>
        AI-Based Food Quality Classification
        & Environmental Analytics
        <br>
        Machine Learning Research Prototype | 2026
    </div>
    """,
    unsafe_allow_html=True
)

   
