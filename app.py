import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="ML Visualizer", layout="centered", initial_sidebar_state="expanded")

# --- FUNCTION: SET BACKGROUND IMAGE ---
def set_background(image_file):
    # Ignoring image_file parameter to use CSS background from bg_animation.txt
    bxs = "0 0 0"
    gap = 3
    coef = -0.3
    for i in range(1, 5):
        bxs += f", {i*gap}rem 0 0 {i*coef}rem, {-i*gap}rem 0 0 {i*coef}rem, 0 {-i*gap}rem 0 {i*coef}rem, 0 {i*gap}rem 0 {i*coef}rem"
        for j in range(1, 5):
            bxs += f", {i*gap}rem {j*gap}rem 0 {i*j*1.5*coef}rem, {i*gap}rem {-j*gap}rem 0 {i*j*1.5*coef}rem, {-i*gap}rem {j*gap}rem 0 {i*j*1.5*coef}rem, {-i*gap}rem {-j*gap}rem 0 {i*j*1.5*coef}rem"
    
    tiles = "".join(['<div class="card__grid-effect-tile"></div>' for _ in range(100)])
    grid_html = f'<div class="card__grid-effect">{tiles}</div>'

    st.markdown(
        f"""
        {grid_html}
        <style>
        /* Animation Grid Styles */
        .card__grid-effect {{
            position: fixed;
            z-index: 0;
            top: 0; left: 0; right: 0; bottom: 0;
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            grid-template-rows: repeat(10, 1fr);
            pointer-events: auto;
        }}
        .card__grid-effect-tile {{
            position: relative;
        }}
        .card__grid-effect-tile:before {{
            content: '';
            color: #ffffff; /* Use white or slightly visible color for contrast */
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            height: 0.3rem;
            width: 0.3rem;
            border-radius: 50%;
            background: #ffffff;
            transition: 500ms linear all;
            box-shadow: {bxs};
        }}
        .card__grid-effect-tile:hover:before {{
            height: 2rem;
            width: 2rem;
            transition: 70ms linear all;
        }}
        
        /* Ensure app container sits over the background grid and is clickable */
        [data-testid="stAppViewBlockContainer"] {{
            position: relative;
            z-index: 10;
            background: transparent;
            border-radius: 20px;
            padding: 30px;
        }}
        [data-testid="stSidebar"] {{
            background-color: #f0f2f6;
            border-right: 1px solid #d1d5db;
            box-shadow: 2px 0 10px rgba(0,0,0,0.05);
        }}
        /* Permanently open sidebar */
        [data-testid="collapsedControl"] {{
            display: none !important;
        }}
        [data-testid="stSidebar"] * {{
            color: black !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background-color: #939393 !important;
            color: black !important;
            border: 1px dashed #ccc !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: #ccd1d6 !important;
            color: black !important;
            border-radius: 6px;
            border: 1px solid #aaa;
        }}
        .stButton>button {{
            background: transparent !important;
            border-radius: 2px !important;
            border: 2px solid #000000 !important;
            color: black !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            box-shadow: 0px 0px 0px 0px rgba(0, 0, 0, 0.04) !important;
            transition: all .3s, box-shadow .2s, transform .2s .2s !important;
            text-transform: uppercase !important;
        }}
        .stButton>button:hover {{
            color: white !important;
            background: black !important;
            box-shadow: 0px 17px 18px -14px rgba(0, 0, 0, 0.08) !important;
        }}
        [data-testid="stAppViewContainer"] {{
            background-color: #A9C9FF;
            background-image: linear-gradient(180deg, #A9C9FF 0%, #FFBBEC 100%);
            background-size: 100% 100%;
            background-attachment: fixed;
        }}
        /* Removed dark overlay to show gradient clearly */
        [data-testid="stAppViewContainer"] * {{
            color: black;
        }}
        h1 {{ color: black; }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- SET BACKGROUND ---
set_background("bg.jpeg")

# --- TITLE ---
st.title("Lightweight ML Model Visualizer")

# --- UPLOAD CSV ---
st.sidebar.header("Upload Dataset")
st.sidebar.markdown("Upload your structured CSV data here to train and evaluate ML models.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())

    st.subheader("Drop Unnecessary Columns")
    drop_cols = st.multiselect("Select columns to drop (optional)", df.columns)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        st.success(f"Dropped columns: {drop_cols}")

    st.subheader("Select Target Column")
    target_col = st.selectbox("Choose target/output column", df.columns)

    st.subheader("Preprocessing")
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.success("Missing values dropped.")

    # Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    st.info("ℹ️ Categorical features encoded.")

    # Encode target if categorical
    y = df[target_col]
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        st.info("Target column encoded.")

    X = df.drop(columns=[target_col])

    st.subheader("📑 Final Feature & Target Shape")
    st.write("🟩 Features (X):", X.shape)
    st.write("Target (y):", y.shape)

    # --- MODEL SELECTION ---
    st.subheader("🤖 Select & Train Model")

    model_name = st.radio(
        "Choose model:",
        (
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes"
        )
    )
    test_size = st.slider("Test size (split ratio)", 0.1, 0.5, 0.2)

    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # --- MODEL INITIALIZATION ---
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42, n_estimators=100)
        elif model_name == "Support Vector Machine (SVM)":
            model = SVC(kernel="rbf", probability=True)
        elif model_name == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naive Bayes":
            model = GaussianNB()

        # --- TRAINING ---
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- METRICS ---
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Model trained successfully! Accuracy: **{acc:.2f}**")

        # --- CONFUSION MATRIX ---
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(set(y))),
            yticks=range(len(set(y))),
            xticklabels=set(y),
            yticklabels=set(y),
            ylabel='True label',
            xlabel='Predicted label',
            title="Confusion Matrix"
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="black")
        st.pyplot(fig)

        # --- CLASSIFICATION REPORT ---
        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # --- FEATURE IMPORTANCE / COEFFICIENTS ---
        st.subheader("Feature Importance / Coefficients")

        if model_name in ["Decision Tree", "Random Forest"]:
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(importance)
        elif model_name == "Logistic Regression":
            importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
            st.bar_chart(importance)
        else:
            st.info("ℹ️ Feature importance is not available for this model type.")

else:
    st.warning("👈 Please upload a CSV file to begin.")
