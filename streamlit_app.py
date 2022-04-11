# Imports
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("model_dataset.csv")

st.title("Accidents In France")
st.image("./streamlit files/accident.jpg", width = 700)

st.sidebar.header("Pages to view")
st.sidebar.selectbox("", ["The Project", "Visualize The Data", "Make Prediction", "About"])

# Display dataframe
st.write("## The dataset")
col_names = df.columns.tolist()
st.dataframe(df[st.multiselect("Columns:", col_names,
                               default=["accident_severity", "lighting", "intersection", "atmosphere"])])






# Visualize the dataset by count
st.write("## Visualize the data")
visualize_by = st.selectbox("Visualize by:", col_names)

fig = plt.figure(figsize=(10, 6))
plt.style.use("dark_background")
sns.countplot(df[visualize_by], palette ="tab20")
plt.xticks(rotation=20)
plt.xlabel('')
plt.ylabel('')
st.pyplot(fig)


# Map visuals
df_map = pd.read_csv("streamlit_map_data.csv")
df_map['latitude']=pd.to_numeric(df_map['lat']) 
df_map['longitude']=pd.to_numeric(df_map['lon'])
st.write("### Geographical visuals")
st.map(df_map)







# Making predictions
st.write("## Predict accident severity")

#model = pickle.load(open("model.pkl", "rb"))

st.subheader("User Input Parameters")


# Function to get user input
def get_user_input():
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
prediction = 'Not Fatal' #model.predict(input_df)

# Display prediction
st.write(f"### The maximum accident severity that can occur from the given conditions is {prediction}")