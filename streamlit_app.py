# Imports
import pandas as pd
import streamlit as st
#import pickel
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("model_dataset.csv")

st.title("Accidents In France")
#st.image("https://storage.googleapis.com/kaggle-datasets-images/24824/31630/a5f5ce1e4b4066d1f222e79e8286f077/dataset-cover.jpg?t=2018-05-03-00-52-48", width = 700)
st.write("""Get an overview of""")