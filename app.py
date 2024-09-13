import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import warnings
from transformers import pipeline

# Suppress specific future warnings from the transformers module
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

# Suppress DeprecationWarnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load the dataset
df = pd.read_csv('data/processed-mobile-data.csv')

def main():
    
    st.title('Mobile Recommendation System')
    st.sidebar.title('Navigation')

    # Sidebar navigation
    options = st.sidebar.radio('Select an Option', ['Home', 'Recommendations', 'Visualizations'])
    
    if options == 'Home':
        show_home()
    elif page == 'Recommendations':
        show_recommendations()
    elif page == 'Visualizations':
        show_visualizations()
    elif options == 'Chat with AI':
        chat_with_ai()

def show_home():
    st.header('Welcome to the Mobile Recommendation System')
    st.write('This app helps you find the best mobile phones based on various criteria.')
    st.image('background.jpg', use_column_width=True)

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
        if filtered_df.empty:
            filtered_df = df[(df['Price'] >= min_price) & 
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




