# -*- coding: utf-8 -*-
"""MLcollege.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a0m6kklL4sVVkCHnhXD4GhnZTYklTBS5
"""

#!pip install streamlit_lottie

#!pip install streamlit
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from streamlit_lottie import st_lottie
import requests
import json

st.set_page_config(page_title="Exoplanet Mass Predictor", layout="wide")

# ----------------- Load Animation -----------------
def load_lottie_url(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None


# ----------------- Load Data -----------------
@st.cache_data

def load_data():
    # Corrected URL
    url = "https://raw.githubusercontent.com/studentofstars/Mini-Project-continued/main/Exoplanet_data.csv"
    df = pd.read_csv(url)
    df = df.drop(columns=['Unnamed: 0']).drop_duplicates()
    return df

# Load animations# Load
lottie_space_intro = load_lottie_url("https://raw.githubusercontent.com/studentofstars/Mini-Project-continued/main/Animation%20-%201745037899411.json")

lottie_success = load_lottie_url("https://raw.githubusercontent.com/studentofstars/Mini-Project-continued/main/Animation%20-%201745038127530.json")


# ----------------- Page Title -----------------
st_lottie(lottie_space_intro, height=250, key="space_intro")
st.title("🚀 Exoplanet Mass & Similarity Explorer")

# ----------------- Data & Model -----------------
data = load_data()
X = data[['pl_orbper', 'pl_orbsmax', 'st_mass']]
y = data['pl_bmasse']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(random_state=42)
model.fit(X_scaled, y)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📊 Visualizations", "🪐 Similar Planet", "📄 Dataset"])

# ================= TAB 1: PREDICTION =================
with tab1:
    st.header("🔮 Predict Planet Mass")
    st.markdown("""
   

    Enter the planet's orbital characteristics and its star's mass, and the app will estimate the mass of the exoplanet using a trained machine learning model (Random Forest Regressor).

    - 🌌 *Orbital Period*: Time the planet takes to orbit its star.
    - 🌍 *Semi-Major Axis*: Average distance from the star.
    - ☀️ *Star Mass*: Mass of the host star (in Solar masses).

    The model will then:
    - Predict the planet's mass (in Earth masses).
    - Categorize it as lighter than Earth, between Earth and Jupiter, or heavier (gas giant).
    - Let you download your prediction as a CSV file.
    """)
    orbper = st.number_input("Orbital Period (days)", min_value=0.0, step=0.1, value=1.0)
    orbsmax = st.number_input("Semi-Major Axis (AU)", min_value=0.0, step=0.01, value=0.01)
    starmass = st.number_input("Star Mass (Solar Mass)", min_value=0.0, step=0.01, value=1.0)

    input_features = np.array([[orbper, orbsmax, starmass]])
    input_scaled = scaler.transform(input_features)
    predicted_mass = model.predict(input_scaled)[0]

    st_lottie(lottie_success, height=150, key="success")
    st.success(f"🌍 Predicted Planet Mass: **{predicted_mass:.2f} Earth masses**")

    earth_mass = 1
    jupiter_mass = 317.8
    if predicted_mass < earth_mass:
        st.info("🟢 Lighter than Earth!")
    elif predicted_mass < jupiter_mass:
        st.info("🟡 Between Earth and Jupiter")
    else:
        st.info("🔴 Heavier than Jupiter! Possibly a gas giant.")

    # Export Prediction
    #st.subheader("📥 Export Prediction")
    #if st.button("Download Prediction as CSV"):
        #"""pred_df = pd.DataFrame({
            #"Orbital Period": [orbper],
            #"Semi-Major Axis": [orbsmax],
            #"Star Mass": [starmass],
           # "Predicted Planet Mass": [predicted_mass]
       # })
        #st.download_button("Download", pred_df.to_csv(index=False), file_name="prediction.csv")"""

# ================= TAB 2: VISUALIZATIONS =================
with tab2:
    st.header("📊 Explore the Dataset")
    st.markdown("""

    Visualize different aspects of the exoplanet dataset to understand trends and relationships between features.

    **You can explore:**
    - 📉 *Mass Distribution*: See how planet masses are spread across the dataset.
    - 🔁 *Feature Correlation*: Heatmap showing how features relate to each other.
    - 🌐 *Orbital Period vs Mass*: Understand how orbital time impacts mass.
    - 🔍 *Feature Importance*: See which features most influence the model’s predictions.
    """)

    plot_choice = st.selectbox("Choose a Plot", ["Mass Distribution", "Feature Correlation", "Orbital Period vs Mass"])

    if plot_choice == "Mass Distribution":
        fig = px.histogram(data, x='pl_bmasse', nbins=30,
                           title="Distribution of Planet Masses",
                           labels={'pl_bmasse': "Planet Mass (Earth Masses)"},
                           color_discrete_sequence=['#4e79a7'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### 🌍 Planet Mass Distribution  
        This plot shows the **distribution of planet masses** (in Earth masses) in the dataset.  
        It helps us understand how common small, medium, and large exoplanets are.  

        - Peaks in the histogram represent the most common mass ranges.
        - You might notice a concentration of Earth-like or Jupiter-like planets.

        Useful for identifying trends in planetary formation.
        """)

    elif plot_choice == "Feature Correlation":
        corr = data.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr,
                        text_auto=True,
                        title="Correlation Heatmap",
                        color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
          ### 🔁 Feature Correlation Heatmap  
          This heatmap displays the **correlation coefficients** between all numeric parameters in the dataset.

          - Values range from `-1` (strong negative correlation) to `+1` (strong positive correlation).
          - Blue indicates a positive relationship; red indicates a negative one.
          - Use this to see which features are strongly related and might influence each other.

          **📌 Parameters involved:**

          - `pl_orbper`: Orbital period of the planet (in Earth days)  
          - `pl_orbsmax`: Semi-major axis — the average distance between the planet and its star (in AU)  
          - `st_mass`: Mass of the host star (in solar masses)  
          - `pl_bmasse`: Planet mass (in Earth masses)

         These are the core physical parameters used to train the ML model and analyze exoplanet properties.
         """)


    elif plot_choice == "Orbital Period vs Mass":
        fig = px.scatter(data, x='pl_orbper', y='pl_bmasse',
                         title="Orbital Period vs Planet Mass",
                         labels={'pl_orbper': "Orbital Period (days)", 'pl_bmasse': "Planet Mass (Earth Masses)"},
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### ⏱️ Orbital Period vs Planet Mass  
        This scatter plot shows how **planet mass relates to orbital period** (time it takes to orbit the star).

        - Shorter periods may indicate close-in, possibly smaller planets.
        - Longer periods might suggest massive planets on wider orbits.

        It's a good way to visualize any clusters or outliers in planetary systems.
        """)

    # Feature Importance Plot
    st.subheader("🔍 Feature Importance")
    st.markdown("""
    The bar chart below shows which input features (orbital period, semi-major axis, and star mass) had the **most influence** on the machine learning model when predicting planet mass.

    This can help us decide which parameters are most crucial for accurate predictions.
    """)
    importances = model.feature_importances_
    feature_names = X.columns
    fig = px.bar(x=feature_names, y=importances,
                 labels={'x': 'Feature', 'y': 'Importance'},
                 title="Which Features Influence Mass Prediction?")
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: SIMILAR PLANET =================
with tab3:
    st.header("🪐 Find a Similar Known Exoplanet")
    st.markdown("""
    After you enter the planet characteristics in the Prediction tab, this section will:

    - Calculate the **most similar known exoplanet** from the dataset using Euclidean distance in feature space.
    - Show you its characteristics for comparison.

    Great for discovering planets that resemble the one you imagined!
    """)

    distances = euclidean_distances(X_scaled, input_scaled)
    nearest_index = np.argmin(distances)
    similar_planet = data.iloc[nearest_index]

    st.write("Most similar known planet:")
    st.dataframe(similar_planet.to_frame().T)

# ================= TAB 4: DATASET =================
with tab4:
    st.header("📄 Dataset Preview")
    st.markdown("""
    This tab displays the full dataset used to train the model.

    You can:
    - Browse all the planetary entries.
    - See the raw values of features like orbital period, semi-major axis, star mass, and planet mass.

    Transparency helps build trust in machine learning predictions!
    """)
    st.dataframe(data)




