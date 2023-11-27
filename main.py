from flask import Flask, request, jsonify
from utils.prep import generate_products_json
from utils.pinecone import upsert_doc
from utils.model import generate_response_df, generate_evaluated_response_df

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


@app.route('/generate-response', methods=['POST'])
def generateResponse():
    if request.method == 'POST':
        response_df = generate_response_df(request)
        response_obj = response_df.to_json(orient='records')
    return response_obj


@app.route('/generate-response-with-score', methods=['POST'])
def generateResponseWithScore():
    if request.method == 'POST':
        response_df = generate_evaluated_response_df(request)
        response_obj = response_df.to_json(orient='records')
    return response_obj


if __name__ == '__main__':
    app.run(debug=True, port=3000)
