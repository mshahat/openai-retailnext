"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import ast

import json
from IPython.display import Image, display, HTML

import main as backend


# streamlit layout

# ---- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="RetailNext — Visual Stylist",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('My First App')
st.write("Here's our first attempt at using data to create a table:")

df = pd.DataFrame({
'first column': [1, 2, 3, 4],
'second column': [10, 20, 30, 40]
})

df

def get_embeddings():
    # MOHAMED - Read Catalog of Clothes items
    styles_filepath = "data/sample_clothes/sample_styles.csv"
    styles_df = pd.read_csv(styles_filepath, on_bad_lines='skip')
    print(styles_df.head())
    print("Opened dataset successfully. Dataset has {} items of clothing.".format(len(styles_df)))

    # MOHAMED - Not Generating own embeddings for demo, commented the following
    # generate_embeddings(styles_df, 'productDisplayName')
    # print ("Writing embeddings to file ...")
    # styles_df.to_csv('data/sample_clothes/sample_styles_with_embeddings.csv', index=False)
    # print ("Embeddings successfully stored in sample_styles_with_embeddings.csv")

    # MOHAMED - Use already generated embeddings for demo
    styles_df = pd.read_csv('data/sample_clothes/sample_styles_with_embeddings.csv', on_bad_lines='skip')

    # Convert the 'embeddings' column from string representations of lists to actual lists of floats
    styles_df['embeddings'] = styles_df['embeddings'].apply(lambda x: ast.literal_eval(x))

    print(styles_df.head())
    print("Opened dataset successfully. Dataset has {} items of clothing along with their embeddings.".format(len(styles_df)))

    return styles_df

# RetailNext cookbook routine

# ------------------------
# MOHAMED - this block is borrowed from main.py

# Set the path to the images and select a test image
image_path = "data/sample_clothes/sample_images/"
test_images = ["2133.jpg", "7143.jpg", "4226.jpg"]

# Encode the test image to base64
reference_image = image_path + test_images[0]
encoded_image = backend.encode_image_to_base64(reference_image)

styles_df = get_embeddings()

# Select the unique subcategories from the DataFrame
unique_subcategories = styles_df['articleType'].unique()

# Analyze the image and return the results
analysis = backend.analyze_image(encoded_image, unique_subcategories)
image_analysis = json.loads(analysis)

# Display the image and the analysis results
display(Image(filename=reference_image))
print(image_analysis)

# Extract the relevant features from the analysis
item_descs = image_analysis['items']
item_category = image_analysis['category']
item_gender = image_analysis['gender']

# Filter data such that we only look through the items of the same gender (or unisex) and different category
filtered_items = styles_df.loc[styles_df['gender'].isin([item_gender, 'Unisex'])]
filtered_items = filtered_items[filtered_items['articleType'] != item_category]
print(str(len(filtered_items)) + " Remaining Items")

# Find the most similar items based on the input item descriptions
matching_items = backend.find_matching_items_with_rag(filtered_items, item_descs)

# Display the matching items (this will display 2 items for each description in the image analysis)
html = ""
paths = []
for i, item in enumerate(matching_items):
    item_id = item['id']

    # Path to the image file
    image_path = f'data/sample_clothes/sample_images/{item_id}.jpg'
    paths.append(image_path)
    html += f'<img src="{image_path}" style="display:inline;margin:1px"/>'

# Print the matching item description as a reminder of what we are looking for
print(item_descs)
# Display the image
display(HTML(html))

# Select the unique paths for the generated images
paths = list(set(paths))

for path in paths:
    # Encode the test image to base64
    suggested_image = backend.encode_image_to_base64(path)

    raw = backend.check_match(encoded_image, suggested_image)
    print("RAW RESPONSE START >>>", repr(raw[:3000]), "<<< END")

    if raw:
        # Check if the items match
        match = json.loads(raw)

        # Display the image and the analysis results
        if match["answer"] == 'yes':
            display(Image(filename=path))
            print("The items match!")
            print(match["reason"])

# ------------------------


