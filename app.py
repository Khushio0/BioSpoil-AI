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

# Fruit image sources
FRUIT_IMAGES = {
    "Banana": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=900&q=85",
    "Orange": "https://images.unsplash.com/photo-1547514701-42782101795e?w=900&q=85",
    "Pineapple": "https://images.unsplash.com/photo-1550258987-190a2d41a8ba?w=900&q=85",
    "Tomato": "https://images.unsplash.com/photo-1546094096-0df4bcaaa337?w=900&q=85"
}

# =====================================================
# 2. PROFESSIONAL THEME AND CSS
# =====================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #F4F7F3;
    color: #20352B;
}

/* Main content */
section[data-testid="stMain"] {
    background-color: #F4F7F3;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #174B37 !important;
    font-weight: 800 !important;
}

p, li, label {
    color: #35483E;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #123D30 0%,
        #176448 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.35);
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #E0EEE5 !important;
}

/* Sidebar radio */
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

div[data-testid="stMetricLabel"] {
    color: #53685B !important;
}

div[data-testid="stMetricValue"] {
    color: #174B37 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #176B4A !important;
    color: #FFFFFF !important;
    border: 1px solid #176B4A !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #104B34 !important;
    border-color: #104B34 !important;
    color: #FFFFFF !important;
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

/* Selectbox text */
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

/* Dataframes */
[data-testid="stDataFrame"] {
    border: 1px solid #D6E4D9;
    border-radius: 12px;
}

/* Alerts */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Fruit images */
div[data-testid="stImage"] img {
    object-fit: contain !important;
    background-color: #FFFFFF !important;
    border-radius: 13px !important;
    padding: 8px !important;
}

/* Fruit card containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #D5E2D8 !important;
    border-radius: 15px !important;
    background-color: #FFFFFF;
}

/* Divider */
hr {
    border-color: #D5E1D8 !important;
}

/* Links */
a {
    color: #176B4A !important;
}

/* Sidebar footer */
.sidebar-footer {
    color: #E3F0E7;
    font-size: 13px;
    line-height: 1.7;
}

/* Hero */
.hero {
    background: linear-gradient(
        115deg,
        #174B37 0%,
        #247A59 60%,
        #70A98A 100%
    );
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

/* Fruit descriptions */
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

/* Responsive layout */
@media (max-width: 768px) {
    .hero {
        padding: 24px;
    }

    .hero h1 {
        font-size: 27px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.is_file():

        matching = [
            f for f in BASE_DIR.iterdir()
            if f.is_file()
            and f.name.lower() == MODEL_FILENAME.lower()
        ]

        if matching:
            model_path = matching[0]
        else:
            raise FileNotFoundError(
                f"{MODEL_FILENAME} was not found "
                "in the application directory."
            )

    else:
        model_path = MODEL_PATH

    return joblib.load(model_path)


try:
    model = load_model()

except Exception as e:

    st.error(f"Unable to load trained model: {e}")

    st.info(
        "Please check that the trained .pkl file "
        "is uploaded alongside app.py in GitHub."
    )

    st.stop()


# =====================================================
# 4. HEADER
# =====================================================

st.markdown("""
<div style="margin-bottom:20px;">

<h1 style="
color:#174B37;
font-size:38px;
font-weight:900;
margin-bottom:3px;
">
🍃 BioSpoil AI
</h1>

<p style="
color:#53685B;
font-size:16px;
font-weight:500;
">
AI-Based Food Quality Classification & Environmental Analytics
</p>

</div>
""", unsafe_allow_html=True)


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("""
<div style="padding:12px 0 20px 0;">

<h2 style="color:white !important;">
🍃 BioSpoil AI
</h2>

<p style="
color:#E3F0E7 !important;
font-size:14px;
">
Food Quality Intelligence
</p>

</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

st.sidebar.markdown("### NAVIGATION")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Single Prediction",
        "Batch Analysis",
        "About the Model"
    ],
    label_visibility="collapsed"
)

st.sidebar.divider()

st.sidebar.markdown("""
<div class="sidebar-footer">

<strong>Technology Stack</strong><br>
Python • Scikit-learn<br>
Pandas • Plotly • Streamlit

<br><br>

<strong>Project Type</strong><br>
Machine Learning Research Prototype

<br><br>

©️ BioSpoil AI | 2026

</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. PREDICTION FUNCTION
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
# 7. OVERVIEW / HOME PAGE
# =====================================================

if page == "Overview":

    st.markdown("""
    <div class="hero">

        <h1>🍃 Welcome to BioSpoil AI</h1>

        <h3>
        Smarter Food Quality Classification
        Through Artificial Intelligence
        </h3>

        <p>
        Explore how environmental conditions can help
        predict food quality using Machine Learning.
        </p>

        <p>
        An interactive research prototype combining
        environmental parameters and predictive analytics
        to classify food samples.
        </p>

    </div>
    """, unsafe_allow_html=True)

    # Platform statistics

    st.markdown("## 📊 Platform at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Food Categories", "04")
    c2.metric("Prediction Classes", "02")
    c3.metric("Input Parameters", "05")
    c4.metric("ML Algorithm", "RF")

    st.divider()

    # Fruit gallery

    st.markdown("## 🍎 Explore Food Categories")

    st.write(
        "Discover the four food categories analysed "
        "by the BioSpoil AI classification model."
    )

    fruit_columns = st.columns(4)

    descriptions = {
        "Banana": "Tropical fruit",
        "Orange": "Citrus fruit",
        "Pineapple": "Tropical fruit",
        "Tomato": "Fresh produce"
    }

    for col, fruit in zip(fruit_columns, FRUITS):

        with col:

            with st.container(border=True):

                st.image(
                    FRUIT_IMAGES[fruit],
                    use_container_width=True
                )

                st.markdown(
                    f'<div class="fruit-name">{fruit}</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="fruit-description">'
                    f'{descriptions[fruit]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.divider()

    # Environmental intelligence

    st.markdown("## 🌍 Environmental Intelligence")

    st.write(
        "Explore the environmental features used "
        "by the machine learning model."
    )

    parameter_information = {

        "Food Type": (
            "Fruit",
            "Categorical",
            "Identifies the food category being analysed."
        ),

        "Temperature": (
            "Temp",
            "Numerical (°C)",
            "Represents the surrounding temperature."
        ),

        "Humidity": (
            "Humid (%)",
            "Numerical (%)",
            "Represents the relative humidity."
        ),

        "Light Intensity": (
            "Light (Fux)",
            "Numerical",
            "Represents the light measurement in the dataset."
        ),

        "CO₂": (
            "CO2 (pmm)",
            "Numerical",
            "Represents the CO2 measurement in the dataset."
        )
    }

    selected_parameter = st.selectbox(
        "Select a parameter to explore",
        list(parameter_information.keys())
    )

    feature, data_type, explanation = (
        parameter_information[selected_parameter]
    )

    st.markdown(
        f"""
        <div style="
        background:#E5F0E7;
        border-left:5px solid #247A59;
        padding:20px;
        border-radius:10px;
        margin-top:10px;
        ">

        <h4 style="color:#174B37;">
        {feature} | {data_type}
        </h4>

        <p style="color:#35483E;">
        {explanation}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # How it works

    st.markdown("## ⚙️ How BioSpoil AI Works")

    w1, w2, w3 = st.columns(3)

    with w1:

        with st.container(border=True):

            st.markdown(
                '<div class="workflow-number">01</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="workflow-title">📥 Input</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="workflow-description">
                Enter food type and environmental
                measurements.
                </div>
                """,
                unsafe_allow_html=True
            )

    with w2:

        with st.container(border=True):

            st.markdown(
                '<div class="workflow-number">02</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="workflow-title">⚙️ Analyse</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="workflow-description">
                The trained Random Forest model
                processes the input features.
                </div>
                """,
                unsafe_allow_html=True
            )

    with w3:

        with st.container(border=True):

            st.markdown(
                '<div class="workflow-number">03</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="workflow-title">📊 Predict</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="workflow-description">
                Obtain a predicted Good/Bad class
                and model confidence score.
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # Quick prediction

    st.markdown("## 🔬 Try BioSpoil AI")

    st.write(
        "Enter a sample's environmental conditions "
        "and explore the model prediction."
    )

    with st.form("homepage_prediction"):

        col1, col2 = st.columns(2)

        with col1:

            home_fruit = st.selectbox(
                "Select Food Category",
                FRUITS,
                key="home_fruit"
            )

            home_temp = st.number_input(
                "Temperature (°C)",
                value=24.0,
                key="home_temp"
            )

            home_humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0,
                key="home_humidity"
            )

        with col2:

            home_light = st.number_input(
                "Light (dataset units)",
                value=12.5,
                key="home_light"
            )

            home_co2 = st.number_input(
                "CO₂ (dataset units)",
                value=370.0,
                key="home_co2"
            )

        home_submit = st.form_submit_button(
            "Analyse Food Quality",
            use_container_width=True
        )

    if home_submit:

        home_sample = pd.DataFrame([{
            "Fruit": home_fruit,
            "Temp": home_temp,
            "Humid (%)": home_humidity,
            "Light (Fux)": home_light,
            "CO2 (pmm)": home_co2
        }])

        try:

            home_result = predict_data(home_sample)

            prediction = str(
                home_result["Predicted Class"].iloc[0]
            )

            confidence = home_result[
                "Model Confidence (%)"
            ].iloc[0]

            st.markdown("### Prediction Result")

            rc1, rc2 = st.columns(2)

            rc1.metric("Predicted Class", prediction)

            rc2.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            if prediction.strip().lower() == "good":

                st.success(
                    "The model classifies this sample as GOOD."
                )

            elif prediction.strip().lower() == "bad":

                st.error(
                    "The model classifies this sample as BAD."
                )

            else:

                st.info(f"Predicted class: {prediction}")

            scores = pd.DataFrame({
                "Class": model.classes_,
                "Score": (
                    model.predict_proba(home_sample)[0] * 100
                )
            })

            fig = px.bar(
                scores,
                x="Class",
                y="Score",
                color="Class",
                text_auto=".2f",
                title="Prediction Probability"
            )

            fig.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#20352B",
                yaxis_range=[0, 100]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.warning(
                "Model confidence is not a direct measure "
                "of food safety or microbial spoilage."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")

    st.divider()

    st.info(
        "BioSpoil AI is a research prototype. "
        "Its predictions represent dataset-defined "
        "classification labels and require external "
        "validation before practical food-safety use."
    )


# =====================================================
# 8. SINGLE PREDICTION
# =====================================================

elif page == "Single Prediction":

    st.markdown("## 🔬 Individual Food Classification")

    st.write(
        "Enter the environmental conditions to "
        "obtain a machine learning prediction."
    )

    with st.form("single_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            fruit = st.selectbox(
                "Select Food",
                FRUITS
            )

            temp = st.number_input(
                "Temperature (°C)",
                value=24.0
            )

            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0
            )

        with col2:

            light = st.number_input(
                "Light (dataset units)",
                value=12.5
            )

            co2 = st.number_input(
                "CO₂ (dataset units)",
                value=370.0
            )

        submitted = st.form_submit_button(
            "Analyse Food Sample",
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

            prediction = str(
                result["Predicted Class"].iloc[0]
            )

            confidence = result[
                "Model Confidence (%)"
            ].iloc[0]

            st.divider()

            st.markdown("### 📊 Prediction Results")

            if prediction.strip().lower() == "good":

                st.success(
                    "Predicted Classification: GOOD"
                )

            elif prediction.strip().lower() == "bad":

                st.error(
                    "Predicted Classification: BAD"
                )

            else:

                st.info(f"Predicted Classification: {prediction}")

            c1, c2 = st.columns(2)

            c1.metric("Predicted Class", prediction)

            c2.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            score_columns = [
                f"{label} Score (%)"
                for label in model.classes_
            ]

            probability_data = pd.DataFrame({
                "Class": model.classes_,
                "Model Score (%)": [
                    result[column].iloc[0]
                    for column in score_columns
                ]
            })

            fig = px.bar(
                probability_data,
                x="Class",
                y="Model Score (%)",
                color="Class",
                title="Predicted Class Scores",
                text_auto=".2f"
            )

            fig.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font_color="#20352B"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.warning(
                "Model confidence is not a measure of "
                "actual food safety or microbial spoilage."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =====================================================
# 9. BATCH ANALYSIS
# =====================================================

elif page == "Batch Analysis":

    st.markdown("## 📂 Batch Food Classification")

    st.write(
        "Upload a CSV file containing the five "
        "required model input features."
    )

    st.markdown("### Required CSV Columns")

    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file:

        try:

            batch = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Dataset")

            st.dataframe(
                batch.head(),
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

                st.warning("The uploaded CSV is empty.")

            elif st.button(
                "Run Batch Prediction",
                use_container_width=True
            ):

                results = predict_data(batch)

                st.markdown("### Classification Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.markdown("### Class Distribution")

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
                    title="Predicted Food Classification",
                    hole=0.4
                )

                fig.update_layout(
                    paper_bgcolor="#FFFFFF",
                    font_color="#20352B"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.metric(
                    "Total Samples Analysed",
                    len(results)
                )

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇ Download Prediction Report",
                    csv,
                    "BioSpoil_AI_Results.csv",
                    "text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(f"Batch analysis error: {e}")


# =====================================================
# 10. ABOUT THE MODEL
# =====================================================

elif page == "About the Model":

    st.markdown("## 🧠 Machine Learning Model")

    st.markdown("### Selected Algorithm")

    st.success("Random Forest Classifier")

    st.write(
        """
        Random Forest is an ensemble learning algorithm
        that combines multiple decision trees to perform
        classification.

        It generates predictions by combining the outputs
        of multiple decision trees.
        """
    )

    st.divider()

    st.markdown("### Model Input Features")

    parameter_data = pd.DataFrame({

        "Parameter": [
            "Fruit",
            "Temperature",
            "Humidity",
            "Light",
            "CO₂"
        ],

        "Feature": FEATURES,

        "Type": [
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

    st.markdown("### Prediction Classes")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### 🟢 GOOD")

            st.write(
                "The sample is classified as Good "
                "according to the trained model."
            )

    with c2:

        with st.container(border=True):

            st.markdown("### 🔴 BAD")

            st.write(
                "The sample is classified as Bad "
                "according to the trained model."
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

    st.info(
        "The reported model performance should be "
        "interpreted within the original dataset. "
        "External validation is necessary before "
        "making generalizable claims."
    )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div style="
text-align:center;
padding:18px;
background:#E6EFE7;
border-radius:12px;
">

<h3 style="color:#174B37 !important;">
🍃 BioSpoil AI
</h3>

<p style="color:#35483E;">
AI-Based Food Quality Classification & Environmental Analytics
</p>

<p style="color:#53685B;font-size:13px;">
Machine Learning Research Prototype | 2026
</p>

</div>
""", unsafe_allow_html=True)



 

