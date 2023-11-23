import os
import numpy as np
import openai
import pinecone

from langchain.document_loaders.json_loader import JSONLoader
from langchain.vectorstores.pinecone import Pinecone
from langchain.text_splitter import CharacterTextSplitter

MODEL = "text-embedding-ada-002"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["tags"] = record.get("tags")
    metadata["images_list"] = record.get("images_list")
    metadata["handle"] = record.get("handle")
    return metadata


def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        res = openai.Embedding.create(input=[text], engine=MODEL)
        embeddings_list.append(res['data'][0]['embedding'])
    return embeddings_list

# Define a function to upsert embeddings to Pinecone


def upsert_embeddings_to_pinecone(index, namespace, embeddings, metadatas):
    vectors = []
    for metadata, embedding in zip(metadatas, embeddings):
        vectors.append(
            {'id': str(metadata['id']), 'values': embedding, 'metadata': metadata})
    print(vectors[0])
    index.upsert(vectors=vectors, namespace=namespace)


def upsert_doc():
    file_path = './products.json'
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[]',
        content_key="expanded_description",
        metadata_func=metadata_func
    )
    documents = loader.load_and_split()
    texts = [str(doc) for doc in documents]
    metadatas = [{
        'id': doc.metadata['id'],
        'text': doc.page_content,
        'title': doc.metadata['title'],
        'handle': doc.metadata['handle'],
        'images_list': doc.metadata['images_list']
    } for doc in documents]
    embeddings = create_embeddings(texts)

    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env,
    )
    index = pinecone.Index(pinecone_index_name)

    print('docs:', len(documents))
    print('texts:', len(texts))
    print('embeddings:', len(embeddings))
    print('metadatas:', len(metadatas))
    upsert_embeddings_to_pinecone(
        index, pinecone_namespace, embeddings, metadatas)
    return
