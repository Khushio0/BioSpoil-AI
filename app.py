import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="BioSpoil AI",
    page_icon="🍃",
    layout="wide"
)

# Locate the model in the same directory as app.py
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

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f8f5;
}
.main-title {
    color: #164b36;
    font-size: 38px;
    font-weight: 800;
}
.subtitle {
    color: #60776b;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dce8df;
}
.stButton > button {
    background-color: #176b48;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    global MODEL_PATH

    # Handle capitalization differences in filenames
    if not MODEL_PATH.is_file():

        matching_files = [
            file for file in BASE_DIR.iterdir()
            if file.is_file()
            and file.name.lower() == MODEL_FILENAME.lower()
        ]

        if matching_files:
            MODEL_PATH = matching_files[0]

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file '{MODEL_FILENAME}' was not found. "
            f"Please ensure it is uploaded to your GitHub "
            f"repository in the same folder as app.py. "
            f"Application directory: {BASE_DIR}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Could not load trained model: {e}")

    st.info(
        "Check that BioSpoil_AI_Best_Model.pkl "
        "exists in your GitHub repository alongside app.py."
    )

    st.stop()


# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">🍃 BioSpoil AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Based Food Quality Classification & Environmental Analytics</div>',
    unsafe_allow_html=True
)

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🍃 BioSpoil AI")
st.sidebar.caption("Food Quality Intelligence")

page = st.sidebar.radio(
    "Navigation",
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

# ---------------- PREDICTION FUNCTION ----------------

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


# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.header("Project Overview")

    st.write(
        "BioSpoil AI is a machine learning research prototype "
        "that classifies food samples as Good or Bad using "
        "environmental parameters."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Food Categories", "4")
    c2.metric("Prediction Classes", "2")
    c3.metric("Selected Model", "Random Forest")

    st.subheader("Environmental Parameters")

    st.write(
        "The model uses the following input features:"
    )

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
        use_container_width=True
    )

    st.info(
        "This prototype predicts dataset-defined Good/Bad "
        "labels. It does not directly measure microbial growth "
        "or establish food safety."
    )


# ---------------- SINGLE PREDICTION ----------------

elif page == "Single Prediction":

    st.header("Individual Food Classification")

    st.write(
        "Enter environmental conditions to obtain a model prediction."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            fruit = st.selectbox(
                "Select Food",
                ["Banana", "Orange", "Pineapple", "Tomato"]
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

            prediction = result[
                "Predicted Class"
            ].iloc[0]

            confidence = result[
                "Model Confidence (%)"
            ].iloc[0]

            st.divider()

            st.subheader("Prediction Results")

            if prediction == "Good":
                st.success(
                    "Predicted Classification: GOOD"
                )

            else:
                st.error(
                    "Predicted Classification: BAD"
                )

            col1, col2 = st.columns(2)

            col1.metric(
                "Predicted Class",
                prediction
            )

            col2.metric(
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

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.warning(
                "Model confidence is not a measure of actual "
                "food safety or microbial spoilage."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ---------------- BATCH ANALYSIS ----------------

elif page == "Batch Analysis":

    st.header("Batch Food Classification")

    st.write(
        "Upload a CSV containing the five model input features."
    )

    st.code(", ".join(FEATURES))

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file:

        batch = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset")

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

        elif st.button("Run Batch Prediction"):

            try:

                results = predict_data(batch)

                st.subheader("Classification Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                st.subheader("Class Distribution")

                counts = results[
                    "Predicted Class"
                ].value_counts().reset_index()

                counts.columns = [
                    "Class",
                    "Number of Samples"
                ]

                fig = px.pie(
                    counts,
                    names="Class",
                    values="Number of Samples",
                    title="Predicted Food Classification"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Prediction Report",
                    csv,
                    "BioSpoil_AI_Results.csv",
                    "text/csv"
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )


# ---------------- MODEL INFORMATION ----------------

elif page == "About the Model":

    st.header("Machine Learning Model")

    st.subheader("Selected Algorithm")

    st.success("Random Forest Classifier")

    st.write(
        "Random Forest is an ensemble learning algorithm "
        "that combines multiple decision trees to perform "
        "classification."
    )

    st.subheader("Model Information")

    st.write("**Input features:**")

    for feature in FEATURES:
        st.write(f"- {feature}")

    st.write("**Output classes:**")

    st.write("- Good")
    st.write("- Bad")

    st.subheader("Model Architecture")

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
        "The reported model performance should be interpreted "
        "within the original dataset. External validation is "
        "needed before making generalizable claims."
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "BioSpoil AI | Research Prototype | "
    "Machine Learning-Based Food Quality Classification"
)



      

    
   

      
