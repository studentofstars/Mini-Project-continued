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

uploaded_file = "http://raw.githubusercontent.com/studentofstars/Mini-Project-continued/main/data_labels_cleaned.csv"

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
    
     # Label encoding
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Train/test split
    X = df.drop(columns=['Drought phase classification:'])
    y = df['Drought phase classification:']
    test_size = st.slider("Select Test Size", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create tabs
    tabs = st.tabs(["📊 Dataset Overview", "📈 Visualization", "🤖 Model Training", "📌 Feature Importance", "🔮 Predict", "🗃️ Raw Data"])

    # Dataset Overview Tab
    with tabs[0]:
        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape}")
        st.dataframe(df.head())
        
     # Visualization Tab
    with tabs[1]:
        st.subheader("Correlation Heatmap")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(), annot=False, fmt=".2f", cmap='coolwarm', ax=ax_corr)
        st.pyplot(fig_corr)

     

        if 'latitude' in df.columns and 'longitude' in df.columns:
            st.subheader("🗺️ Spatial Distribution of Drought Records")
            fig_map = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                                        color='Drought phase classification:',
                                        mapbox_style='carto-positron', zoom=4)
            st.plotly_chart(fig_map)
            
     # Model Training Tab
    with tabs[2]:
        st.subheader("Model Performance")
        st.write("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)
        
    # Feature Importance Tab
    with tabs[3]:
        st.subheader("Feature Importances")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots()
        sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax_imp)
        st.pyplot(fig_imp)
    
    # Prediction Tab
    with tabs[4]:
        st.subheader("Enter Input for Prediction")
        input_data = {}
        with st.form("prediction_form"):
            for feature in X.columns:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                input_data[feature] = st.slider(f"{feature}", min_value=min_val, max_value=max_val, value=mean_val)
            submitted = st.form_submit_button("Predict")
            if submitted:
                input_df = pd.DataFrame([input_data])
                prediction = model.predict(input_df)[0]
                label = prediction
                if 'Drought phase classification:' in label_encoders:
                    label = label_encoders['Drought phase classification:'].inverse_transform([prediction])[0]
                st.success(f"🌟 Predicted Drought Phase: **{label}**")
                st.balloons()
     # Raw Data Tab
    with tabs[5]:
        st.subheader("Raw Dataset")
        st.dataframe(df)

        st.download_button("Download Raw Data", df.to_csv(index=False).encode('utf-8'), "raw_data.csv", "text/csv")
       
        
else:
    st.info("Please upload a dataset to begin.")

 

  

   
