import os
import numpy as np
import pymongo
from pymongo import MongoClient, UpdateOne

client = MongoClient(
    os.getenv("MONGO_DB_CONNECTION"))
db = client['ConversationLog']
collection = db['Prompts']


def get_documents(query={}):
    return collection.find(query)


def update_documents(df):
    updates = []

    for _, row in df.iterrows():
        updates.append(UpdateOne({'_id': row.get('_id')}, {
                       '$set': {
                           'openai_relevance_0': row.get('openai_relevance_0'),
                           'openai_relevance_1': row.get('openai_relevance_1'),
                           'openai_precision@1': row.get('openai_precision@1'),
                           'openai_precision@2': row.get('openai_precision@2')
                       }}, upsert=True))

    collection.bulk_write(updates)
    return 'ok'
