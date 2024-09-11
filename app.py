import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data/processed-mobile-data.csv')

def main():
    
    st.title('Mobile Recommendation System')
    st.sidebar.title('Navigation')
    
    # Sidebar navigation
    options = st.sidebar.radio('Select an Option', ['Home', 'Recommendations', 'Visualizations'])
    
    if options == 'Home':
        show_home()
    elif options == 'Recommendations':
        show_recommendations()
    elif options == 'Visualizations':
        show_visualizations()

def show_home():
    st.header('Welcome to the Mobile Recommendation System')
    st.write('This app helps you find the best mobile phones based on various criteria.')
    st.image('background.jpg', use_column_width=True)

def show_recommendations():
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
        min_value=10000,  # Minimum value of the slider
        max_value=500000,  # Maximum value of the slider
        value=(10000, 500000),  # Initial values: 10,000 to 500,000
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
        
        # Convert Image URLs to HTML
        filtered_df['Image_HTML'] = filtered_df['Image_URL'].apply(lambda url: f'<img src="{url}" width="100" />')
        
        # Drop the column before 'Name' (assuming it's the 'Index' or any other column)
        if 'Index' in filtered_df.columns:
            filtered_df = filtered_df.drop(columns=['Index'])
        
        # Display Recommendations as a Table
        if not filtered_df.empty:
            st.write('Here are the recommended mobiles:')
            
            # Create HTML table with image
            html = filtered_df.to_html(escape=False, columns=['Name', 'Brand', 'System', 'Storage', 'RAM', 'Processor', 'Ratings', 'Price', 'Image_HTML'])
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.write('No mobiles found with the selected criteria.')

def show_visualizations():
    st.header('Visualizations')
    
    # Ratings Distribution
    st.subheader('Ratings Distribution')
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size for better visibility
    sns.histplot(df['Ratings'], bins=20, ax=ax, color='skyblue', kde=True)
    ax.set_title('Distribution of Ratings', fontsize=16, color='white')
    ax.set_xlabel('Ratings', fontsize=14, color='white')
    ax.set_ylabel('Frequency', fontsize=14, color='white')
    
    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)  # Transparent background for the figure
    ax.spines['top'].set_edgecolor('white')
    ax.spines['right'].set_edgecolor('white')
    ax.spines['left'].set_edgecolor('white')
    ax.spines['bottom'].set_edgecolor('white')
    ax.tick_params(axis='both', colors='white')
    
    st.pyplot(fig)
    
    # Price vs Ratings
    st.subheader('Price vs Ratings')
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust size for better visibility
    scatter = sns.scatterplot(data=df, x='Price', y='Ratings', hue='Brand', palette='pastel', ax=ax)
    ax.set_title('Price vs Ratings', fontsize=16, color='white')
    ax.set_xlabel('Price', fontsize=14, color='white')
    ax.set_ylabel('Ratings', fontsize=14, color='white')
    
    # Add legend to the scatter plot with white background
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, labels, title="Brands", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, title_fontsize='13', facecolor='white', frameon=True)
    
    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)  # Transparent background for the figure
    ax.spines['top'].set_edgecolor('white')
    ax.spines['right'].set_edgecolor('white')
    ax.spines['left'].set_edgecolor('white')
    ax.spines['bottom'].set_edgecolor('white')
    ax.tick_params(axis='both', colors='white')
    
    st.pyplot(fig)
    
    # Improved Brand Distribution Pie Chart
    st.subheader('Brand Distribution')
    fig, ax = plt.subplots(figsize=(10, 7))  # Adjust the size for better visibility
    brand_counts = df['Brand'].value_counts()
    
    # Use a color palette
    colors = sns.color_palette("pastel")[0:len(brand_counts)]
    
    # Create the pie chart without percentage display
    wedges, texts = ax.pie(
        brand_counts,
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.3, edgecolor='w'),  # Make the chart a bit more distinct
        textprops={'fontsize': 12, 'color': 'white'}  # Text color white
    )
    
    # Add a legend to the pie chart with white background
    ax.legend(wedges, brand_counts.index,
              title="Brands",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=12,
              title_fontsize='13',
              facecolor='white',  # White background for the legend
              frameon=True)
    
    # Remove background color and set border color
    ax.set_facecolor('none')
    fig.patch.set_alpha(0.0)  # Transparent background for the figure
    
    ax.set_title('Brand Distribution', fontsize=16, color='white')
    
    st.pyplot(fig)


if __name__ == '__main__':
    main()
