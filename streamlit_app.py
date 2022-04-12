# Imports
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("./streamlit files/model_dataset.csv")

st.title("Accidents In France")


st.sidebar.header("Pages to view")
page = st.sidebar.selectbox("", ["The Project", "Visualize", "Predictions", "About"])



def display_dataframe():
    """Display dataframe"""
    st.image("./streamlit files/accident.jpg", width = 700)
    st.write("## The dataset 📰")
    col_names = df.columns.tolist()
    st.dataframe(df[st.multiselect("Columns:", col_names,
                                   default=["accident_severity", "lighting", "intersection", "atmosphere"])])
    
    
def visualize_data():
    """Visualize the dataset by count"""
    st.write("## Visualize the data 📊")
    
    col_names = df.columns.tolist()
    
    visualize_by = st.selectbox("Visualize by:", col_names)

    fig = plt.figure(figsize=(10, 6))
    plt.style.use("dark_background")
    
    # Visualize with respect to accident severity
    add_hue = st.checkbox("Accident Severity")
    if add_hue:
        sns.countplot(df[visualize_by], hue=df["accident_severity"], palette ="tab20")
    else:
        sns.countplot(df[visualize_by], palette ="tab20")
        
    plt.xticks(rotation=20)
    plt.xlabel('')
    plt.ylabel('')
    st.pyplot(fig)


def visualize_map():
    # Map visuals
    # df_map = pd.read_csv("./streamlit files/streamlit_map_data.csv")
    df_map = pd.read_csv("./streamlit files/streamlit_map_data.csv")
    st.write("### Geographical visuals")
    st.map(df_map)
    
    
def use_model():
    """Making predictions"""
    st.write("## Predict accident severity 📈")
    st.subheader("Input Parameters")

    def get_user_input():
        """Get user input"""
        features = {}

        def add_selectbox(column, title):
            features[column] = st.selectbox(title, df[column].unique())

        add_selectbox('hour', "Hour of the day")
        add_selectbox('lighting', "Lighting condition")
        add_selectbox('atmosphere', "Atmospheric condition")
        add_selectbox('collision', "Type of collision")
        add_selectbox('localisation', "Localisation")
        add_selectbox('user_category', "User category")
        add_selectbox('user_sex', "User sex")
        add_selectbox('pedestrian_action', "Pedestrian action")
        add_selectbox('road_category', "Road category")
        add_selectbox('traff_regime', "Traffic regime")
        add_selectbox('longitud_profile', "Longitudinal profile")
        add_selectbox('drawing_plan', "Drawing plan")
        add_selectbox('surface_cond', "Surface condition")
        add_selectbox('acc_situation', "Accident occurance")

        return pd.DataFrame(features, index=[0])


    input_df = get_user_input()
     
    # Make predicitions
    predict = st.button("Predict 📈")
    if predict:
        #model = pickle.load(open("./streamlit files/dtc_model.pkl", "rb"))
        prediction = 'Not Fatal' #model.predict(input_df)

        st.write(f"##### The maximum accident severity that can occur from the given conditions is {prediction}")
    

    
# Website flow logic    
if page == "The Project":
    display_dataframe()
elif page == "Visualize The Data":
    visualize_data()
    #visualize_map()
elif page == "Make Prediction":
    use_model()
elif page == "About":
    pass