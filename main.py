import pandas as pd
from flask import Flask, request, jsonify

# from utils.format import camel_to_snake
from utils.prep import generate_products_json
from utils.pinecone import upsert_doc, get_vectorstores_data
from utils.model import generate_response_df, evaluate_response_df
from utils.mongodb import get_documents, update_documents
from utils.arizeAI import get_arize_url

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


@app.route('/evaluate-response', methods=['POST'])
def evaluateResponse():
    if request.method == 'POST':
        query_df = pd.DataFrame([request.json])
        response_df = evaluate_response_df(query_df)
        response_obj = response_df.to_json(orient='records')
    return response_obj


@app.route('/evaluate-db-records', methods=['POST'])
def evaluateDBRecords():
    if request.method == 'POST':
        documents = get_documents()
        query_df = pd.DataFrame(list(documents))
        query_df = query_df[query_df.openai_relevance_0.isnull()]

        if (len(query_df) < 1):
            return 'no new records to evaluate'

        response_df = evaluate_response_df(query_df)
        update_documents(response_df)

        # response_obj = response_df.to_json(orient="records")
    return f"{len(query_df)} documents evaluated"


@app.route('/get-phoenix-session', methods=['GET'])
def getArizeAIView():
    if request.method == 'GET':
        database_df = get_vectorstores_data()
        query_df = pd.DataFrame(list(get_documents()))

        session = get_arize_url(query_df, database_df)

        print('session =', session)

    return session.url


if __name__ == '__main__':
    app.run(debug=True, port=3000)
