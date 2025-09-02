"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import ast

from main import generate_embeddings

def init_data():

    # MOHAMED - Read Catalog of Clothes items
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    print(styles_df.head())
    print("Opened dataset successfully. Dataset has {} items of clothing.".format(len(styles_df)))

    # MOHAMED - Not Generating own embeddings for demo, commented the following
    # generate_embeddings(styles_df, 'productDisplayName')
    # print("Writing embeddings to file ...")
    # styles_df.to_csv('data/sample_clothes/sample_styles_with_embeddings.csv', index=False)
    # print("Embeddings successfully stored in sample_styles_with_embeddings.csv")

    # MOHAMED - Use already generated embeddings for demo
    styles_df = pd.read_csv('data/sample_clothes/sample_styles_with_embeddings.csv', on_bad_lines='skip')

    # Convert the 'embeddings' column from string representations of lists to actual lists of floats
    styles_df['embeddings'] = styles_df['embeddings'].apply(lambda x: ast.literal_eval(x))

    print(styles_df.head())
    print("Opened dataset successfully. Dataset has {} items of clothing along with their embeddings.".format(len(styles_df)))

def app():
    st.title('My First App')
    st.write("Here's our first attempt at using data to create a table:")

    df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

    df

init_data()
app()
