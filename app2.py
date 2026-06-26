import streamlit as st
import pickle
import io
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
st.set_page_config(page_title="ML Visualizer", layout="centered")

# --- FUNCTION: SET BACKGROUND IMAGE WITH SEMI-TRANSPARENT PANEL ---
def set_background(image_file):
    import base64
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(f"""
        <style>
        /* === SIDEBAR STYLING === */
        [data-testid="stSidebar"] {{
            background-color: #f8f9fa;
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}

        /* === BACKGROUND IMAGE === */
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* === CENTER GLASS PANEL (FIXED) === */
        [data-testid="stAppViewBlockContainer"] {{
            background: rgba(0, 0, 0, 0.55);      /* translucent black glass */
            backdrop-filter: blur(30px);          /* glass effect */
            -webkit-backdrop-filter: blur(10px);
            border-radius: 25px;
            padding: 40px 60px;
            margin: 60px auto;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
            max-width: 1100px;
        }}

        /* === TEXT VISIBILITY === */
        [data-testid="stAppViewBlockContainer"] * {{
            color: #f5f5f5!important;
        }}
        <h1 style="
        text-align: left;
        color: white;
        font-size: 42px;
        font-weight: 700;
        white-space: nowrap;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
        ">
        🧠 Lightweight ML Model Visual
        </h1>

        # h1, h2, h3, h4, h5, h6 {{
        #     color: #ffffff !important;
        #     text-shadow: 0px 0px 10px rgba(0,0,0,0.7);
        # }}

        # .stDataFrame {{
        #     background-color: rgba(255, 255, 255, 0.15) !important;
        #     border-radius: 10px;
        # }}
        </style>
    """, unsafe_allow_html=True)

# --- APPLY BACKGROUND ---
set_background("bg.jpg")

# # --- START THE CENTER PANEL ---
# st.markdown('<div class="center-panel">', unsafe_allow_html=True)

# --- MAIN TITLE ---
# st.title("🧠 Lightweight ML Model Visualizer")
st.markdown(
    """
    <div style="
        text-align: center;
        color: white;
        font-size: 46px;
        font-weight: 800;
        margin-top: 20px;
        white-space: nowrap;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.6);
    ">
        🧠 Lightweight ML Model Visualizer
    </div>
    """,
    unsafe_allow_html=True
)


# --- UPLOAD CSV ---
st.sidebar.header("📂 Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", df.columns.tolist())

    st.subheader("🧹 Drop Unnecessary Columns")
    drop_cols = st.multiselect("Select columns to drop (optional)", df.columns)
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        st.success(f"✅ Dropped columns: {drop_cols}")

    st.subheader("🎯 Select Target Column")
    target_col = st.selectbox("Choose target/output column", df.columns)

    st.subheader("⚙️ Preprocessing")
    if st.checkbox("Drop rows with missing values"):
        df.dropna(inplace=True)
        st.success("✅ Missing values dropped.")

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    st.info("ℹ️ Categorical features encoded.")

    y = df[target_col]
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        st.info("🎯 Target column encoded.")

    X = df.drop(columns=[target_col])

    st.subheader("📑 Final Feature & Target Shape")
    st.write("🟩 Features (X):", X.shape)
    st.write("🎯 Target (y):", y.shape)

    st.subheader("🤖 Select & Train Model")
    model_name = st.radio("Choose model:", (
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "K-Nearest Neighbors (KNN)",
        "Naive Bayes"
    ))
    test_size = st.slider("Test size (split ratio)", 0.1, 0.5, 0.2)

    if st.button("🚀 Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # --- CHOOSE MODEL ---
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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model in session state
        st.session_state["trained_model"] = model
        st.session_state["X_columns"] = list(X.columns)
        st.session_state["le_target"] = le_target if 'le_target' in locals() else None

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy: **{acc:.2f}**")

        # --- MODEL DOWNLOAD OPTION ---
        buffer = io.BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)

        st.download_button(
            label="💾 Download Trained Model",
            data=buffer,
            file_name=f"{model_name.replace(' ', '_')}_model.pkl",
            mime="application/octet-stream"
        )
        st.info("You can download this trained model as a .pkl file and use it later for predictions!")


        st.subheader("📊 Confusion Matrix")
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

        st.subheader("📋 Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.subheader("📈 Feature Importance / Coefficients")
        if model_name in ["Decision Tree", "Random Forest"]:
            importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.bar_chart(importance)
        elif model_name == "Logistic Regression":
            importance = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)
            st.bar_chart(importance)
        else:
            st.info("ℹ️ Feature importance is not available for this model type.")

                # --- CUSTOM INPUT TESTING ---
        st.subheader("🧪 Test the Model on Custom Input")
        test_choice = st.radio("Do you want to test the trained model with your own data?", ("No", "Yes"))

        if test_choice == "Yes":
            if "trained_model" not in st.session_state:
                st.warning("⚠️ Please train a model first before testing.")
            else:
                model = st.session_state["trained_model"]
                X_cols = st.session_state["X_columns"]
                le_target = st.session_state["le_target"]

                st.info("Enter feature values below:")
                user_input = []
                for feature in X_cols:
                    val = st.text_input(f"Enter value for **{feature}**:")
                    user_input.append(val)

                if st.button("🔍 Predict on Custom Input"):
                    try:
                        cleaned_input = []
                        for v in user_input:
                            if v.strip() == "":
                                cleaned_input.append(0.0)
                            else:
                                cleaned_input.append(float(v))

                        input_df = pd.DataFrame([cleaned_input], columns=X_cols)
                        pred = model.predict(input_df)[0]

                        if le_target is not None:
                            pred = le_target.inverse_transform([pred])[0]

                        st.success(f"🎯 Predicted Class: **{pred}**")

                    except Exception as e:
                        st.error(f"⚠️ Error during prediction: {e}")



st.markdown('</div>', unsafe_allow_html=True)
