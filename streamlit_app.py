# Imports
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

    
df = pd.read_csv("streamlit files/model_dataset.csv")

st.set_page_config(layout="wide")
st.title("Accidents In France")


st.sidebar.header("Pages")
page = st.sidebar.selectbox("", ["Home", "Visualize", "Predict", "About"])



def data_page():
    """Display dataframe"""
    st.image("streamlit files/accident.jpg", width = 700)
    st.write("""
    - Get an overview of accident occurances in France
    
    - Gain insights on how several external conditions affects the severity of an accident
    
    - Use a pretrain model to predict accident severity""")
    
    st.write("## The dataset 📰")
    st.write("""
    The dataset used for the project was gotten from [Kaggle](https://www.kaggle.com/datasets/ahmedlahlou/accidents-in-france-from-2005-to-2016)
    
    It consisted of several CSV files which were merged and preprocessed to the final format shown below""")
    
    col_names = df.columns.tolist()
    st.dataframe(df[st.multiselect("Columns:", col_names,
                                   default=["accident_severity", "lighting", "intersection", "atmosphere"])])
    
    
def visualize_page():
    """Visualize the dataset by count"""
    st.write("## Visualize the data 📊")
    
    col_names = df.columns.tolist()
    
    visualize_by = st.selectbox("Visualize by:", col_names)

    fig = plt.figure(figsize=(10, 4))
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
    # Map visualsv")
    df_map = pd.read_csv("streamlit files/streamlit_map_data.csv")
    st.write("### Geographical visuals")
    st.map(df_map)
    
    
def modelling_page():
    """Making predictions"""
    
    st.write("## Make predictions 📈")
    
    def get_user_input():
        """Get user input"""
        features = {}

        def add_selectbox(position, column, title):
            features[column] = position.selectbox(title, df[column].unique())
        
        expander = st.expander("Input Parameters ", expanded=True)
        col1, col2, col3 = expander.columns(3)
        
        add_selectbox(col1, 'hour', "Hour of the day")
        add_selectbox(col1, 'lighting', "Lighting condition")
        add_selectbox(col1, 'intersection', "Point of intersection")
        add_selectbox(col1, 'atmosphere', "Atmospheric condition")
        add_selectbox(col1, 'collision', "Type of collision")
        add_selectbox(col2, 'localisation', "Localisation")
        add_selectbox(col2, 'user_category', "User category")
        add_selectbox(col2, 'user_sex', "User sex")
        add_selectbox(col2, 'pedestrian_action', "Pedestrian action")
        add_selectbox(col2, 'road_category', "Road category")
        add_selectbox(col3, 'traff_regime', "Traffic regime")
        add_selectbox(col3, 'longitud_profile', "Longitudinal profile")
        add_selectbox(col3, 'drawing_plan', "Drawing plan")
        add_selectbox(col3, 'surface_cond', "Surface condition")
        add_selectbox(col3, 'acc_situation', "Accident occurance")

        return pd.DataFrame(features, index=[0])


    input_df = get_user_input()
    
    pred_col, result_col = st.columns((1,4))
    predict = pred_col.button("Predict 📈")
    
    # Make predicitions
    if predict:
        model = pickle.load(open("streamlit files/dtc_model.pkl", "rb"))
        encoders = pickle.load(open("streamlit files/encoders.pkl", "rb"))
        
        # Encode the features using the same encoder that was used in modelling
        for column in input_df.columns:
            input_df[column] = encoders[column].transform(input_df[column])
        
        # Make predictions
        prediction = model.predict(input_df)
        result = encoders["accident_severity"].inverse_transform(prediction)
        result_col.write(f"##### The maximum accident severity that can occur from the given conditions is {result[0]}")
    

    
def about_page():
    st.write("""
    This was done as the captone project for the HDSC winter '22 Internship
    
    By team XGBoost 🚀
    
    All files used for the project can be found [here](https://github.com/jehddie/Accidents-In-France)
    """)
    
    
    
# Website flow logic    
if page == "Home":
    data_page()
elif page == "Visualize":
    visualize_page()
    #visualize_map()
elif page == "Predict":
    modelling_page()
elif page == "About":
    about_page()