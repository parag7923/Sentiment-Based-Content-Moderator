from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Define the API route for sentiment analysis
@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()

    if 'content' not in data:
        return jsonify({'error': 'No content provided'}), 400
    
    content = data['content']
    
    # Analyze the sentiment of the content
    sentiment_result = sentiment_analyzer(content)

    # Classify content based on sentiment
    sentiment = sentiment_result[0]['label']
    score = sentiment_result[0]['score']

    if sentiment == 'NEGATIVE':
        flagged = True
        message = "Content flagged for negative sentiment."
    else:
        flagged = False
        message = "Content is acceptable."

    # Return the result as JSON
    return jsonify({
        'content': content,
        'sentiment': sentiment,
        'score': score,
        'flagged': flagged,
        'message': message
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
