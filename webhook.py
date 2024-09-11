from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('data/processed-mobile-data.csv')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    intent = data['queryResult']['intent']['displayName']
    
    if intent == 'Recommendation':
        parameters = data['queryResult']['parameters']
        brand = parameters.get('brand')
        storage = parameters.get('storage')
        ram = parameters.get('ram')
        min_price = parameters.get('min_price',10000)
        max_price = parameters.get('max_price')

        filtered_df = df[
            (df['Brand'] == brand) &
            (df['Storage'] == storage) &
            (df['RAM'] == ram) &
            (df['Price'] >= min_price) &
            (df['Price'] <= max_price)
        ]

        filtered_df = df.copy()

        # Apply filters step by step
        if brand:
            filtered_df = filtered_df[filtered_df['Brand'] == brand]

        if storage:
            filtered_df = filtered_df[filtered_df['Storage'] == storage]

        if ram:
            filtered_df = filtered_df[filtered_df['RAM'] == ram]

        # Apply price range filter
        filtered_df = filtered_df[
            (filtered_df['Price'] >= min_price) & 
            (filtered_df['Price'] <= max_price)
        ]

        # If no match, progressively remove conditions and provide fallback
        if filtered_df.empty:
            # Try removing RAM filter
            filtered_df = df[(df['Brand'] == brand) & 
                             (df['Storage'] == storage) & 
                             (df['Price'] >= min_price) & 
                             (df['Price'] <= max_price)]
            
            if filtered_df.empty:
                # Try removing Storage filter
                filtered_df = df[(df['Brand'] == brand) & 
                                 (df['Price'] >= min_price) & 
                                 (df['Price'] <= max_price)]

                if filtered_df.empty:
                    # Fallback to only Price filter
                    filtered_df = df[
                        (df['Price'] >= min_price) & 
                        (df['Price'] <= max_price)
                    ]

                    # Sort by highest rating
                    filtered_df = filtered_df.sort_values(by='Ratings', ascending=False)

       
        if not filtered_df.empty:
            result = filtered_df.iloc[0]['Name']  # Return the first match
            return jsonify({'fulfillmentText': f'We recommend: {result}'})
        else:
            return jsonify({'fulfillmentText': 'No mobiles found with the selected criteria.'})


if __name__ == '__main__':
    app.run(port=5000)
