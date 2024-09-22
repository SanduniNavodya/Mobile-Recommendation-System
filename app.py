import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
import random
import requests
import sys
import os
import pickle
import re

sys.path.append(os.path.abspath('src'))
from remove_ import remove

# Suppress specific future warnings from the transformers module
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Suppress DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set up the page configuration
st.set_page_config(layout="wide")

# Load the dataset
df = pd.read_csv('data/processed-mobile-data.csv')


df1 = pickle.load(file=open(file=r'src/model/dataframe.pkl', mode='rb'))
similarity = pickle.load(file=open(file=r'src/model/similarity.pkl', mode='rb'))

remove()

# Set up the OpenAI text-generation pipeline with clean_up_tokenization_spaces to avoid the warning
# chat_pipeline = pipeline('text-generation', model='gpt2', clean_up_tokenization_spaces=True)

# Define the updated patterns to search in the 'corpus' column
adjusted_patterns = {
    'RAM': r'ram\s?(\d+gb)',  # Handles cases like "ram6gb" or "ram 6gb"
    'Storage': r'storage\s?(\d+gb)',  # Handles "storage128gb" or "storage 128gb"
    'Battery Capacity': r'capacity\s?(\d+\s?mah)',  # Handles "capacity5000mah" or "capacity 5000 mah"
    'Display Size': r'display\s?size\s?(\d+\.\d+\s?cm)',  # Handles "display size16.94cm" or "display size 16.94 cm"
    'Resolution': r'resolution\s?(\d+\s?x\s?\d+\s?pixels)',  # Handles "resolution2400 x 1080 pixels"
    'Processor Type': r'processor\s?type\s?([a-zA-Z0-9\s]+)',  # Handles "processor typequalcomm"
    'Processor Speed': r'processor\s?speed\s?(\d+\.\d+)',  # Handles "processor speed2.6"
    'Camera': r'(\d+mp)',  # Handles cases like "50mp" or "8mp"
    'Network': r'(5g|4g lte)',  # Handles "5g" or "4g lte"
    'Weight': r'weight\s?(\d+g)',  # Handles "weight190g" or "weight 190g"
    'Dimensions': r'dimensions\s?(\d+x\d+x\d+\s?mm)',  # Handles "dimensions160x75x8mm"
    'Operating System': r'system\s?([a-zA-Z0-9\s]+)',  # Handles "systemandroid 12"
}

# Function to extract features from the corpus column
def extract_features(corpus_text, patterns):
    extracted_features = {}
    for feature, pattern in patterns.items():
        match = re.search(pattern, corpus_text.lower())  # Make search case-insensitive
        if match:
            extracted_features[feature] = match.group(1).strip()  # Strip extra spaces
        else:
            extracted_features[feature] = "Not specified"
    return extracted_features

# Apply the function to extract features from the 'corpus' column
df_features = df['corpus'].apply(lambda x: extract_features(x, adjusted_patterns))

# Convert the extracted features into a DataFrame for inspection
df_extracted_features = pd.DataFrame(df_features.tolist())


def main():
    
    # st.title('Mobile Recommendation System')
    st.sidebar.title('Navigation')

    # Sidebar navigation
    options = st.sidebar.radio('Select an Option', ['Home', 'Personalized Picks','Variety Recommendations', 'Visualizations'])
    
    if options == 'Home':
        show_home()
    elif options == 'Personalized Picks':
        show_recommendations()
    elif options == 'Variety Recommendations':
        show_recommendations2()
    elif options == 'Visualizations':
        show_visualizations()


def show_home():
    # Setting the page background color
    st.markdown(
        """
        <style>
        .main {
            background-color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Adding a hero section
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Welcome to Mobile Kings Store</h1>", unsafe_allow_html=True)
    
    # Adding a catchy tagline
    st.markdown("<h3 style='text-align: center; color: #1c6b8c;'>Your one-stop shop for the latest and best smartphones!</h3>", unsafe_allow_html=True)
    
    # Displaying a welcoming image
    st.image('background.jpg', use_column_width=True, caption='Find your perfect mobile phone')

    # Adding a section with a call to action
    st.markdown("<p style='text-align: center; font-size: 18px;'>Let us guide you through our vast selection of mobile phones tailored to your needs. Whether you’re looking for the latest flagship model or a budget-friendly option, we've got you covered!</p>", unsafe_allow_html=True)

    # Adding a button to explore more
    if st.button("Start Exploring"):
        st.write("Navigate to the explore section to browse the latest mobiles!")
    
    # Adding some space
    st.write("\n\n")
    
    # Highlighting categories or offers
    st.subheader("Best Mobile Recommendation App")

# Function to recommend devices based on cosine similarity
def recommend_different_variety(mobile):
    mobile_index = df1[df1['name'] == mobile].index[0]
    similarity_array = similarity[mobile_index]
    different_variety = random.sample(list(enumerate(similarity_array)),k=10)

    recommended_mobiles_variety = []
    recommended_mobiles_IMG_variety = []
    recommended_mobiles_ratings_variety = []
    recommended_mobiles_price_variety = []
    for i in different_variety:
        recommended_mobiles_variety.append(df1['name'].iloc[i[0]])
        recommended_mobiles_IMG_variety.append(fetch_IMG(i[0]))
        recommended_mobiles_ratings_variety.append(df1['ratings'].iloc[i[0]])
        recommended_mobiles_price_variety.append(df1['price'].iloc[i[0]])

    return recommended_mobiles_variety, recommended_mobiles_IMG_variety, recommended_mobiles_ratings_variety, recommended_mobiles_price_variety
    

def recommend(mobile):
    # Make sure to use the same DataFrame for filtering and indexing
    mobile_index = df1[df1['name'] == mobile].index[0]
    similarity_array = similarity[mobile_index]
    
    similar_10_mobiles = sorted(list(enumerate(similarity_array)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_mobiles = []
    recommended_mobiles_IMG = []
    recommended_mobiles_ratings = []
    recommended_mobiles_price = []

    # Iterate and append data using df1 (which has the 'name' column you're filtering on)
    for i in similar_10_mobiles:
        recommended_mobiles.append(df1['name'].iloc[i[0]])
        recommended_mobiles_IMG.append(fetch_IMG(i[0]))
        recommended_mobiles_ratings.append(df1['ratings'].iloc[i[0]])
        recommended_mobiles_price.append(df1['price'].iloc[i[0]])

    return recommended_mobiles, recommended_mobiles_IMG, recommended_mobiles_ratings, recommended_mobiles_price

def show_recommendations2():

    # Add custom CSS to control image size and row spacing
    st.markdown("""
        <style>
        .mobile-image {
            max-width: 150px;
            max-height: 200px;
            object-fit: cover;
            margin-bottom: 10px;
        }
        .recommendation-item {
            border: 1px solid #e0e0e0;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('Mobile Recommender System📲')
    st.markdown('> ##### ***Guide***: Select a mobile phone of your choice from the available options...')
    st.markdown('')

    mobiles = df1['name'].values
    selected_mobile = st.selectbox(label='Select Mobile Name', options=mobiles)

    if st.button('Recommend'):
        # Ensure the values are properly assigned
        recommended_mobiles, mobile_IMG, mobiles_ratings, mobiles_price = recommend(selected_mobile)

        # Clean the price by removing all non-numeric characters
        mobiles_price = [int(re.sub(r'[^\d]', '', str(price))) for price in mobiles_price]

        # Check if recommended_mobiles has enough items to display
        if len(recommended_mobiles) >= 10:
            mobile_name = recommended_mobiles
        else:
            mobile_name = ["N/A"] * 10  # Default if not enough recommendations

        col1, col2, col3, col4, col5 = st.columns(5)
        for i in range(5):
            if i < len(mobile_name):
                with eval(f'col{i+1}'):
                    st.markdown(f"<p style='text-align: center;'>{mobile_name[i]}\n"
                                f"Ratings: {mobiles_ratings[i]}  \n"
                                f"Price: LKR {mobiles_price[i]}", unsafe_allow_html=True)
                    st.image(mobile_IMG[i])

        st.markdown('---')

        mobile_name_variety, mobile_IMG_variety, mobiles_ratings_variety, mobiles_price_variety = recommend_different_variety(selected_mobile)

        st.markdown('## Other Variety of Mobiles')
        st.markdown('---')

        # Clean the price by removing all non-numeric characters
        mobiles_price_variety = [int(re.sub(r'[^\d]', '', str(price))) for price in mobiles_price_variety]

        # Create a grid layout for other varieties of mobiles
        num_cols = 5
        for i in range(0, len(mobile_name_variety), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(mobile_name_variety):
                    with cols[j]:
                        st.markdown(f'<div class="recommendation-item">'
                                    f'<img src="{mobile_IMG_variety[i + j]}" class="mobile-image">'
                                    f'<h3>{mobile_name_variety[i + j]}</h3>'
                                    f'<p>Ratings: {mobiles_ratings_variety[i + j]}</p>'
                                    f'<p>Price: LKR {mobiles_price_variety[i + j]}</p>'
                                    f'</div>', unsafe_allow_html=True)

                        # Expandable section for more details directly under the mobile phone
                        row = df[df['name'] == mobile_name_variety[i + j]]
                        if not row.empty:
                            row = row.iloc[0]
                            if 'corpus' in row:
                                with st.expander(f"View More Details for {row['name']}"):
                                    selected_mobile_row = df[df['name'] == row['name']].iloc[0]
                                    
                                    if 'corpus' in selected_mobile_row:
                                        corpus_text = selected_mobile_row['corpus']
                                        
                                        # Extract features using the function
                                        extracted_features = extract_features(corpus_text, adjusted_patterns)
                                        
                                        # Display the extracted features with appropriate icons
                                        if extracted_features['RAM'] != "Not specified":
                                            st.markdown(f"💾 **RAM:** {extracted_features['RAM']}")
                                        if extracted_features['Storage'] != "Not specified":
                                            st.markdown(f"💽 **Storage:** {extracted_features['Storage']}")
                                        if extracted_features['Battery Capacity'] != "Not specified":
                                            st.markdown(f"🔋 **Battery Capacity:** {extracted_features['Battery Capacity']}")
                                        if extracted_features['Display Size'] != "Not specified":
                                            st.markdown(f"📱 **Display Size:** {extracted_features['Display Size']}")
                                        if extracted_features['Resolution'] != "Not specified":
                                            st.markdown(f"🖥️ **Resolution:** {extracted_features['Resolution']}")
                                        if extracted_features['Processor Type'] != "Not specified":
                                            st.markdown(f"⚙️ **Processor Type:** {extracted_features['Processor Type']}")
                                        if extracted_features['Processor Speed'] != "Not specified":
                                            st.markdown(f"🚀 **Processor Speed:** {extracted_features['Processor Speed']} GHz")
                                        if extracted_features['Camera'] != "Not specified":
                                            st.markdown(f"📷 **Camera:** {extracted_features['Camera']}")
                                        if extracted_features['Network'] != "Not specified":
                                            st.markdown(f"📶 **Network:** {extracted_features['Network']}")
                                        if extracted_features['Weight'] != "Not specified":
                                            st.markdown(f"⚖️ **Weight:** {extracted_features['Weight']}")
                                        if extracted_features['Dimensions'] != "Not specified":
                                            st.markdown(f"📏 **Dimensions:** {extracted_features['Dimensions']}")
                                        if extracted_features['Operating System'] != "Not specified":
                                            st.markdown(f"🖥️ **Operating System:** {extracted_features['Operating System']}")

                                    else:
                                        st.write("No details available.")
                        else:
                            st.write("No details available.")

def fetch_IMG(mobile_index):
    # response = requests.get(url=df['imgURL'].iloc[mobile_index])
    return df1['imgURL'].iloc[mobile_index] 
                  


def show_recommendations():
    # Add custom CSS to control image size and row spacing
    st.markdown("""
        <style>
        .mobile-image {
            max-width: 150px;
            max-height: 200px;
            object-fit: cover;
            margin-bottom: 10px;
        }
        .recommendation-item {
            border: 1px solid #e0e0e0;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.header('Mobile Recommendations')

    # Filter by Brand
    brand = st.selectbox('Select a Brand', df['brand'].unique())  # Ensure this matches your DataFrame's column name
    # Filter by Storage
    storage = st.selectbox('Select Storage Size', sorted(df['storage'].unique()))
    # Filter by RAM
    ram = st.selectbox('Select RAM Size', sorted(df['ram'].unique()))
    # Filter by Price Range with sliders
    min_price, max_price = st.slider(
        'Select Price Range',
        min_value=10000,
        max_value=500000,
        value=(10000, 500000),
        step=1000
    )
    # Add a button to trigger the recommendation display
    if st.button('Show Recommendations'):
         # Apply filters
        filtered_df = df[
            (df['brand'] == brand) &
            (df['storage'] == storage) &
            (df['ram'] == ram) &
            (df['price'] >= min_price) &
            (df['price'] <= max_price)
        ]
        # Handle fallback recommendations
        if filtered_df.empty:
            filtered_df = df[(df['brand'] == brand) & 
                             (df['storage'] == storage) & 
                             (df['price'] >= min_price) & 
                             (df['price'] <= max_price)]
        if filtered_df.empty:
            filtered_df = df[(df['brand'] == brand) & 
                             (df['price'] >= min_price) & 
                             (df['price'] <= max_price)]

        # Display Recommendations as a Grid
        num_cols = 4
        num_rows = (len(filtered_df) + num_cols - 1) // num_cols

        for row_index in range(num_rows):
            cols = st.columns(num_cols)   # Create columns for the current row
            for col_index in range(num_cols):
                item_index = row_index * num_cols + col_index
                if item_index < len(filtered_df):
                    row = filtered_df.iloc[item_index]
                    with cols[col_index]:
                        st.markdown(f'<div class="recommendation-item">'
                                    f'<img src="{row["imgURL"]}" class="mobile-image">'
                                    f'<h3>{row["name"]}</h3>'
                                    f'<p>Brand: {row["brand"]}</p>'
                                    f'<p>Storage: {row["storage"]}GB</p>'
                                    f'<p>RAM: {row["ram"]}GB</p>'
                                    f'<p>Price: LKR {row["price"]}</p>'
                                    f'</div>', unsafe_allow_html=True)
                        
                                              
                        # Expandable section for more details
                        if 'corpus' in row:
                            with st.expander(f"View More Details for {row['name']}"):
                                selected_mobile_row = df[df['name'] == row['name']].iloc[0]
                                
                                if 'corpus' in selected_mobile_row:
                                    corpus_text = selected_mobile_row['corpus']
                                    
                                    # Extract features using the function
                                    extracted_features = extract_features(corpus_text, adjusted_patterns)
                                    
                                   # Display the extracted features with appropriate icons
                                    if extracted_features['RAM'] != "Not specified":
                                        st.markdown(f"💾 **RAM:** {extracted_features['RAM']}")
                                    if extracted_features['Storage'] != "Not specified":
                                        st.markdown(f"💽 **Storage:** {extracted_features['Storage']}")
                                    if extracted_features['Battery Capacity'] != "Not specified":
                                        st.markdown(f"🔋 **Battery Capacity:** {extracted_features['Battery Capacity']}")
                                    if extracted_features['Display Size'] != "Not specified":
                                        st.markdown(f"📱 **Display Size:** {extracted_features['Display Size']}")
                                    if extracted_features['Resolution'] != "Not specified":
                                        st.markdown(f"🖥️ **Resolution:** {extracted_features['Resolution']}")
                                    if extracted_features['Processor Type'] != "Not specified":
                                        st.markdown(f"⚙️ **Processor Type:** {extracted_features['Processor Type']}")
                                    if extracted_features['Processor Speed'] != "Not specified":
                                        st.markdown(f"🚀 **Processor Speed:** {extracted_features['Processor Speed']} GHz")
                                    if extracted_features['Camera'] != "Not specified":
                                        st.markdown(f"📷 **Camera:** {extracted_features['Camera']}")
                                    if extracted_features['Network'] != "Not specified":
                                        st.markdown(f"📶 **Network:** {extracted_features['Network']}")
                                    if extracted_features['Weight'] != "Not specified":
                                        st.markdown(f"⚖️ **Weight:** {extracted_features['Weight']}")
                                    if extracted_features['Dimensions'] != "Not specified":
                                        st.markdown(f"📏 **Dimensions:** {extracted_features['Dimensions']}")
                                    if extracted_features['Operating System'] != "Not specified":
                                        st.markdown(f"🖥️ **Operating System:** {extracted_features['Operating System']}")

                                    # Add more extracted features as needed

                                else:
                                    st.write("No details available.")

        # If there are no recommendations found
        if filtered_df.empty:
            st.write('No mobiles found with the selected criteria.')

def show_visualizations():
    st.header('Visualizations')

    # Ratings Distribution
    st.subheader('Ratings Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['ratings'], bins=20, ax=ax, color='skyblue', kde=True)
    ax.set_title('Distribution of Ratings', fontsize=16, color='white')
    ax.set_xlabel('Ratings', fontsize=14, color='white')
    ax.set_ylabel('Frequency', fontsize=14, color='white')

    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_edgecolor('white')
    ax.spines['right'].set_edgecolor('white')
    ax.spines['left'].set_edgecolor('white')
    ax.spines['bottom'].set_edgecolor('white')
    ax.tick_params(axis='both', colors='white')
    st.pyplot(fig)

    # Price vs Ratings
    st.subheader('Price vs Ratings')
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(data=df, x='price', y='ratings', hue='brand', palette='pastel', ax=ax)
    ax.set_title('Price vs Ratings', fontsize=16, color='white')
    ax.set_xlabel('Price', fontsize=14, color='white')
    ax.set_ylabel('Ratings', fontsize=14, color='white')

    # Add legend to the scatter plot with white background
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, labels, title="Brands", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, title_fontsize='13', facecolor='white', frameon=True)

    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.spines['top'].set_edgecolor('white')
    ax.spines['right'].set_edgecolor('white')
    ax.spines['left'].set_edgecolor('white')
    ax.spines['bottom'].set_edgecolor('white')
    ax.tick_params(axis='both', colors='white')
    st.pyplot(fig)

    # Improved Brand Distribution Pie Chart
    st.subheader('Brand Distribution')
    fig, ax = plt.subplots(figsize=(10, 7))
    brand_counts = df['brand'].value_counts()

    # Use a color palette
    colors = sns.color_palette("pastel")[0:len(brand_counts)]

    # Create the pie chart without percentage display
    wedges, texts = ax.pie(
        brand_counts,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.3, edgecolor='w'),
        textprops={'fontsize': 12, 'color': 'white'}
    )

    # Add a legend to the pie chart with white background
    ax.legend(wedges, brand_counts.index,
              title="Brands",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12,
              title_fontsize='13',
              facecolor='white',
              frameon=True)
    
    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax.set_title('Brand Distribution', fontsize=16, color='white')
    st.pyplot(fig)

if __name__ == '__main__':
    main()