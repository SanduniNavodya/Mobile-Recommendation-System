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
        min_price = parameters.get('min_price')
        max_price = parameters.get('max_price')

        filtered_df = df[
            (df['Brand'] == brand) &
            (df['Storage'] == storage) &
            (df['RAM'] == ram) &
            (df['Price'] >= min_price) &
            (df['Price'] <= max_price)
        ]
        
        if not filtered_df.empty:
            result = filtered_df.iloc[0]['Name']  # Return the first match
            return jsonify({'fulfillmentText': f'We recommend: {result}'})
        else:
            return jsonify({'fulfillmentText': 'No mobiles found with the selected criteria.'})

if __name__ == '__main__':
    app.run(port=5000)
