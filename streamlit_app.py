# Imports
import pandas as pd
import streamlit as st
#import pickel
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("model_dataset.csv")

st.title("Accidents In France")
st.image("./streamlit files/accident.jpg", width = 700)

# Display dataframe
st.write("""Data""")
col_names = df.columns.tolist()
st.dataframe(df[st.multiselect("Columns to display", col_names, default=["accident_severity"])])