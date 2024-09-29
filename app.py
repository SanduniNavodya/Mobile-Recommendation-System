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
df = pd.read_csv(r"C:\\Users\\nimas\\OneDrive\\Documents\\GitHub\\Mobile-Recommendation-System\data\\mobile_processed_data.csv")



df1 = pickle.load(file=open(file=r'src/model/dataframe.pkl', mode='rb'))
similarity = pickle.load(file=open(file=r'src/model/similarity.pkl', mode='rb'))

remove()



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
    if mobile not in df1['name'].values:
        return {
            "names": ["N/A"] * 10,
            "images": [""] * 10,
            "ratings": ["N/A"] * 10,
            "operating_systems": ["N/A"] * 10,
            "prices": ["N/A"] * 10,
            "storage": ["N/A"] * 10,
            "ram": ["N/A"] * 10,
            "processors": ["N/A"] * 10,
            "processor_speeds": ["N/A"] * 10,
            "battery_capacities": ["N/A"] * 10,
            "display_sizes": ["N/A"] * 10,
            "cameras": ["N/A"] * 10,
            "networks": ["N/A"] * 10
        }

    mobile_index = df1[df1['name'] == mobile].index[0]
    similarity_array = similarity[mobile_index]

    # Select random indices for recommendations
    different_variety = random.sample(list(enumerate(similarity_array)), k=10)

    # Initialize lists to hold recommended mobile attributes
    recommended_mobiles_variety = []
    recommended_mobiles_IMG_variety = []
    recommended_mobiles_ratings_variety = []
    recommended_mobiles_operating_system_variety = []
    recommended_mobiles_price_variety = []
    recommended_mobiles_storage_variety = []
    recommended_mobiles_ram_variety = []
    recommended_mobiles_processor_variety = []
    recommended_mobiles_processor_speed_variety = []
    recommended_mobiles_battery_capacity_variety = []
    recommended_mobiles_display_size_variety = []
    recommended_mobiles_camera_variety = []
    recommended_mobiles_network_variety = []

    for i in different_variety:
        recommended_mobiles_variety.append(df1['name'].iloc[i[0]])
        recommended_mobiles_IMG_variety.append(fetch_IMG(i[0]))
        recommended_mobiles_ratings_variety.append(df1['ratings'].iloc[i[0]])
        recommended_mobiles_operating_system_variety.append(df1['operating_system'].iloc[i[0]])
        recommended_mobiles_price_variety.append(df1['price'].iloc[i[0]])

        # Fetch additional specifications
        recommended_mobiles_storage_variety.append(df1['storage'].iloc[i[0]])
        recommended_mobiles_ram_variety.append(df1['ram'].iloc[i[0]])
        recommended_mobiles_processor_variety.append(df1['processor'].iloc[i[0]])
        recommended_mobiles_processor_speed_variety.append(df1['processor_speed'].iloc[i[0]])
        recommended_mobiles_battery_capacity_variety.append(df1['battery_capacity'].iloc[i[0]])
        recommended_mobiles_display_size_variety.append(df1['display_size'].iloc[i[0]])
        recommended_mobiles_camera_variety.append(df1['camera'].iloc[i[0]])
        recommended_mobiles_network_variety.append(df1['network'].iloc[i[0]])

    return {
        "names": recommended_mobiles_variety,
        "images": recommended_mobiles_IMG_variety,
        "ratings": recommended_mobiles_ratings_variety,
        "operating_systems": recommended_mobiles_operating_system_variety,
        "prices": recommended_mobiles_price_variety,
        "storage": recommended_mobiles_storage_variety,
        "ram": recommended_mobiles_ram_variety,
        "processors": recommended_mobiles_processor_variety,
        "processor_speeds": recommended_mobiles_processor_speed_variety,
        "battery_capacities": recommended_mobiles_battery_capacity_variety,
        "display_sizes": recommended_mobiles_display_size_variety,
        "cameras": recommended_mobiles_camera_variety,
        "networks": recommended_mobiles_network_variety
    }

def recommend(mobile):
    mobile_index = df1[df1['name'] == mobile].index[0]
    similarity_array = similarity[mobile_index]

    similar_10_mobiles = sorted(list(enumerate(similarity_array)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_mobiles = []
    recommended_mobiles_IMG = []
    recommended_mobiles_ratings = []
    recommended_mobiles_operating_system_variety = []
    recommended_mobiles_price = []

    for i in similar_10_mobiles:
        recommended_mobiles.append(df1['name'].iloc[i[0]])
        recommended_mobiles_IMG.append(fetch_IMG(i[0]))
        recommended_mobiles_ratings.append(df1['ratings'].iloc[i[0]])
        recommended_mobiles_operating_system_variety.append(df1['operating_system'].iloc[i[0]])
        recommended_mobiles_price.append(df1['price'].iloc[i[0]])

    return recommended_mobiles, recommended_mobiles_IMG, recommended_mobiles_ratings, recommended_mobiles_operating_system_variety, recommended_mobiles_price

def show_recommendations2():
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
    st.markdown('> ##### **Guide**: Select a mobile phone of your choice from the available options...')
    st.markdown('')

    mobiles = df1['name'].values
    selected_mobile = st.selectbox(label='Select Mobile Name', options=mobiles)

    if st.button('Recommend'):
        recommended_mobiles, mobile_IMG, mobiles_ratings, recommended_mobiles_operating_system_variety, mobiles_price = recommend(selected_mobile)

        mobiles_price = [int(re.sub(r'[^\d]', '', str(price))) for price in mobiles_price]

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

        for i in range(5, 10):
            if i < len(mobile_name):
                with eval(f'col{i-4}'):  # Reuse the columns in a new row
                    st.markdown(f"<p style='text-align: center;'>{mobile_name[i]}\n"
                                f"Ratings: {mobiles_ratings[i]}  \n"
                                f"Price: LKR {mobiles_price[i]}", unsafe_allow_html=True)
                    st.image(mobile_IMG[i])
                    

        st.markdown('---')

        # Other varieties of mobiles
        recommendations_variety = recommend_different_variety(selected_mobile)
        
        mobile_name_variety = recommendations_variety["names"]
        mobile_IMG_variety = recommendations_variety["images"]
        mobiles_ratings_variety = recommendations_variety["ratings"]
        recommended_mobiles_operating_system_variety = recommendations_variety["operating_systems"]
        mobiles_price_variety = recommendations_variety["prices"]

        st.markdown('## Other Variety of Mobiles')
        st.markdown('---')

        if 'mobiles_price_variety' not in locals():
            mobiles_price_variety = [0] * len(mobile_name_variety)
        else:
            mobiles_price_variety = [int(re.sub(r'[^\d]', '', str(price))) for price in mobiles_price_variety]

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
                                    f'<p>Operating System: {recommended_mobiles_operating_system_variety[i + j]}</p>'
                                    f'<p>Price: LKR {mobiles_price_variety[i + j]}</p>'
                                    f'</div>', unsafe_allow_html=True)

                        row = df[df['name'] == mobile_name_variety[i + j]]
                        if not row.empty:
                            row = row.iloc[0]
                            if 'corpus' in row:
                                with st.expander(f"View More Details for {row['name']}"):
                                    selected_mobile_row = df[df['name'] == row['name']].iloc[0]
                        else:
                            st.write("No details available.")

def fetch_IMG(mobile_index):
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
    brand = st.selectbox('Select a Brand', df['brand'].unique())

    # Filter by Storage
    storage = st.selectbox('Select Storage Size (GB)', sorted(df['storage'].unique()))

    # Filter by RAM
    ram = st.selectbox('Select RAM Size (GB)', sorted(df['ram'].unique()))

    # Filter by Processor
    processor = st.selectbox('Select Processor', sorted(df['processor'].unique()))

    # Filter by Processor Speed
    processor_speed = st.selectbox('Select Processor Speed (GHz)', sorted(df['processor_speed'].unique()))

    # Filter by Battery Capacity
    battery_capacity = st.slider('Select Battery Capacity (mAh)', 
                                  min_value=int(df['battery_capacity'].min()), 
                                  max_value=int(df['battery_capacity'].max()), 
                                  value=(int(df['battery_capacity'].min()), int(df['battery_capacity'].max())), 
                                  step=100)

    # Filter by Display Size
    display_size = st.selectbox('Select Display Size (cm)', sorted(df['display_size'].unique()))

    # Filter by Camera Quality
    camera_quality = st.selectbox('Select Camera Quality (MP)', sorted(df['camera'].unique()))

    # Filter by Network Type
    network = st.selectbox('Select Network Type', sorted(df['network'].unique()))

    # Filter by Price Range with sliders
    min_price, max_price = st.slider(
        'Select Price Range',
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].min()), int(df['price'].max())),
        step=1000
    )

    # Add a button to trigger the recommendation display
    if st.button('Show Recommendations'):
        # Apply filters
        filtered_df = df[
            (df['brand'] == brand) &
            (df['storage'] == storage) &
            (df['ram'] == ram) &
            (df['processor'] == processor) &
            (df['processor_speed'] == processor_speed) &
            (df['battery_capacity'] >= battery_capacity[0]) &
            (df['battery_capacity'] <= battery_capacity[1]) &
            (df['display_size'] == display_size) &
            (df['camera'] == camera_quality) &
            (df['network'] == network) &
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
                                    f'<p>Storage: {row["storage"]} GB</p>'
                                    f'<p>RAM: {row["ram"]} GB</p>'
                                    f'<p>Processor: {row["processor"]}</p>'
                                    f'<p>Processor Speed: {row["processor_speed"]} GHz</p>'
                                    f'<p>Battery: {row["battery_capacity"]} mAh</p>'
                                    f'<p>Display Size: {row["display_size"]} </p>'
                                    f'<p>Camera: {row["camera"]} MP</p>'
                                    f'<p>Network: {row["network"]}</p>'
                                    f'<p>Price: LKR {row["price"]}</p>'
                                    f'</div>', unsafe_allow_html=True)

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