import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
# from transformers import pipeline
import random
from src.remove_ import remove
import pickle
import re

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

def main():
    
    # st.title('Mobile Recommendation System')
    st.sidebar.title('Navigation')

    # Sidebar navigation
    options = st.sidebar.radio('Select an Option', ['Home', 'Recommendations','Recommendations2', 'Visualizations'])
    
    if options == 'Home':
        show_home()
    elif options == 'Recommendations':
        show_recommendations()
    elif options == 'Recommendations2':
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
    st.title('Mobile Recommender System📲')
    st.markdown('> ##### ***Guide***: :choose Select a mobile phone of your choice from the available options...')
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

        st.markdown('## Other Variety of mobiles')
        st.markdown('---')

        # Clean the price by removing all non-numeric characters
        mobiles_price_variety = [int(re.sub(r'[^\d]', '', str(price))) for price in mobiles_price_variety]

        col11, col12, col13, col14, col15 = st.columns(5)
        for i in range(10):
            with eval(f'col{11 + (i // 5)}'):
                if i < len(mobile_name_variety):
                    st.markdown(f"<p style='text-align: center;'>{mobile_name_variety[i]}\n"
                                f"Ratings: {mobiles_ratings_variety[i]}  \n"
                                f"Price: LKR {mobiles_price_variety[i]}", unsafe_allow_html=True)
                    st.image(mobile_IMG_variety[i])

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
    brand = st.selectbox('Select a Brand', df['Brand'].unique())
    # Filter by Storage
    storage = st.selectbox('Select Storage Size', sorted(df['Storage'].unique()))
    # Filter by RAM
    ram = st.selectbox('Select RAM Size', sorted(df['RAM'].unique()))
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
            (df['Brand'] == brand) &
            (df['Storage'] == storage) &
            (df['RAM'] == ram) &
            (df['Price'] >= min_price) &
            (df['Price'] <= max_price)
        ]
        # Handle fallback recommendations
        if filtered_df.empty:
            filtered_df = df[(df['Brand'] == brand) & 
                             (df['Storage'] == storage) & 
                             (df['Price'] >= min_price) & 
                             (df['Price'] <= max_price)]
        if filtered_df.empty:
            filtered_df = df[(df['Brand'] == brand) & 
                             (df['Price'] >= min_price) & 
                             (df['Price'] <= max_price)]

        # Display Recommendations as a Grid
        num_cols = 4
        num_rows = (len(filtered_df) + num_cols - 1) // num_cols

        for row_index in range(num_rows):
            cols = st.columns(num_cols)   # Create columns for the curren
            for col_index in range(num_cols):
                item_index = row_index * num_cols + col_index
                if item_index < len(filtered_df):
                    row = filtered_df.iloc[item_index]
                    with cols[col_index]:
                        st.markdown(f'<div class="recommendation-item">'
                                    f'<img src="{row["Image_URL"]}" class="mobile-image">'
                                    f'<h3>{row["Name"]}</h3>'
                                    f'<p><strong>Brand:</strong> {row["Brand"]}</p>'
                                    f'<p><strong>Storage:</strong> {row["Storage"]} GB</p>'
                                    f'<p><strong>RAM:</strong> {row["RAM"]} GB</p>'
                                    f'<p><strong>Ratings:</strong> {row["Ratings"]} ⭐</p>'
                                    f'<p><strong>Price:</strong> {row["Price"]} LKR</p>'
                                    f'</div>', unsafe_allow_html=True)
                        
                        if st.button(f"View More Details {row['Name']}", key=item_index):
                            st.write(f"**Processor**: {row['Processor']}")
                            st.write(f"**Operating System**: {row['System']}")
                            st.write(f"**Brand**: {row['Brand']}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        # End item wrapper

        # If there are no recommendations found
        if filtered_df.empty:
            st.write('No mobiles found with the selected criteria.')

def show_visualizations():
    st.header('Visualizations')

    # Ratings Distribution
    st.subheader('Ratings Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Ratings'], bins=20, ax=ax, color='skyblue', kde=True)
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
    scatter = sns.scatterplot(data=df, x='Price', y='Ratings', hue='Brand', palette='pastel', ax=ax)
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
    brand_counts = df['Brand'].value_counts()

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




