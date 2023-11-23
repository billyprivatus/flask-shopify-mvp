from flask import Flask, request, jsonify
from utils.prep import generate_products_json
from utils.pinecone import upsert_doc
from utils.model import generate_evaluated_response

app = Flask(__name__)


@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        status = generate_products_json(request)

        if status != 'success':
            return

        upsert_doc()

        response = {'message': 'Webhook received successfully'}
        return jsonify(response), 200


@app.route('/calculate-response-with-score', methods=['POST'])
def calculateResponseWithScore():
    if request.method == 'POST':
        response_df = generate_evaluated_response(request)
        response_obj = response_df.to_json(orient='records')
    return response_obj


@app.route('/test', methods=['GET'])
def test():
    if request.method == 'GET':
        response = {'message': 'Test successfully'}
        return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, port=3000)
