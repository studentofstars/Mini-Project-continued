import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.set_page_config(page_title="Drought Prediction App", layout="wide")
st.title("🌾 Drought Phase Prediction using Random Forest")

uploaded_file = http://raw.githubusercontent.com/studentofstars/Mini-Project-continued/main/data_labels_cleaned.csv

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop irrelevant columns
    drop_cols = ['Unnamed: 0', '_id', '_uuid', '_submission_time', '_validation_status',
                 '_notes', '_status', '_submitted_by', '_tags', '_index']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    st.subheader("📊 Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    # Drop rows with missing target
    df = df.dropna(subset=['Drought phase classification:'])
    df.fillna(method='ffill', inplace=True)

 

  

    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Train/test split and model
    X = df.drop(columns=['Drought phase classification:'])
    y = df['Drought phase classification:']

    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("✅ Model Performance")
    st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    st.subheader("🔍 Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)

    st.subheader("📌 Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances.values, y=importances.index, ax=ax_imp, palette="viridis")
    ax_imp.set_title("Feature Importances")
    st.pyplot(fig_imp)

    # Optional: Show map if coordinates present
    if 'latitude' in df.columns and 'longitude' in df.columns:
        st.subheader("🗺️ Spatial Distribution of Drought Records")
        fig_map = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                                    color='Drought phase classification:',
                                    mapbox_style='carto-positron',
                                    zoom=4,
                                    title="Geolocation of Drought Phases")
        st.plotly_chart(fig_map)

    if st.checkbox("Show raw data"):
        st.dataframe(df)

else:
    st.info("Please upload a dataset to begin.")
