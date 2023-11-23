import os
import time
import numpy as np
import pandas as pd
import pinecone

from wrapper.openAiEmbeddingWrapper import OpenAIEmbeddingsWrapper
from wrapper.pineconeWrapper import PineconeWrapper

from utils.evaluation import evaluate_retrievals, process_binary_responses

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

MODEL = "text-embedding-ada-002"

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env,
)
index = pinecone.Index(pinecone_index_name)

chat_model_name = "gpt-3.5-turbo"
evaluation_model_name = "gpt-3.5-turbo"
num_retrieved_documents = 2

embeddings = OpenAIEmbeddingsWrapper(model=MODEL)
docsearch = PineconeWrapper.from_existing_index(
    index_name=pinecone_index_name,
    embedding=embeddings,
    namespace=pinecone_namespace
)
llm = ChatOpenAI(model_name=chat_model_name, request_timeout=30)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(
        search_kwargs={"k": num_retrieved_documents}),
)


def generate_query_df(prompts, chain, docsearch, embeddings, num_retrieved_documents=2):
    query_df_data = {"text": [], "text_vector": [],
                     "response": [], "user_feedback": []}

    for context_index in range(num_retrieved_documents):
        query_df_data[f"context_text_{context_index}"] = []
        query_df_data[f"context_similarity_{context_index}"] = []

    for index, prompt in enumerate(prompts):
        print('progress => ', index, '/', len(prompts))
        time.sleep(1)  # Solve hang issue

        response_text = chain.run(prompt)
        print('.')
        retrievals_df = docsearch.retrieval_dataframe.tail(
            num_retrieved_documents)
        print('..')
        contexts = retrievals_df["document_text"].to_list()
        scores = retrievals_df["score"].to_list()
        query_embedding = embeddings.query_embedding_dataframe["text_vector"].iloc[-1]
        print('...')

        query_df_data["text"].append(prompt)
        query_df_data["text_vector"].append(query_embedding.tolist())
        query_df_data["response"].append(response_text)
        query_df_data["user_feedback"].append(float("nan"))  # TODO: Implement

        for context_index in range(num_retrieved_documents):
            context_text = contexts[context_index] if context_index < len(
                contexts) else ""
            score = scores[context_index] if context_index < len(
                scores) else float(0)

            query_df_data[f"context_text_{context_index}"].append(context_text)
            query_df_data[f"context_similarity_{context_index}"].append(score)

    return pd.DataFrame(query_df_data)


def generate_evaluated_response(request):
    num_retrieved_documents = 2

    request_json = request.json
    prompt = request_json['prompt']
    query_df = generate_query_df(
        prompts=[prompt],
        chain=chain,
        docsearch=docsearch,
        embeddings=embeddings,
        num_retrieved_documents=num_retrieved_documents
    )

    context_text_column_names = list(map(
        lambda context_index: f"context_text_{context_index}", range(num_retrieved_documents)))
    openai_evaluations_df = query_df.copy(
    )[["text"] + context_text_column_names]

    for context_index in range(num_retrieved_documents):
        retrievals_data = {
            row["text"]: row[f"context_text_{context_index}"] for _, row in openai_evaluations_df.iterrows()
        }
        raw_responses = evaluate_retrievals(
            retrievals_data, evaluation_model_name)
        processed_responses = process_binary_responses(
            raw_responses, {0: "irrelevant", 1: "relevant"})
        openai_evaluations_df[f"openai_relevance_{context_index}"] = processed_responses

    query_and_evaluations_df = pd.merge(query_df, openai_evaluations_df, on=[
        "text"] + context_text_column_names)
    query_and_evaluations_df[["text", "context_text_0",
                              "context_text_1", "openai_relevance_0", "openai_relevance_1"]]

    num_relevant_documents_array = np.zeros(len(query_and_evaluations_df))

    for retrieved_document_index in range(0, num_retrieved_documents):
        num_retrieved_documents = retrieved_document_index + 1

        num_relevant_documents_array += (
            query_and_evaluations_df[f"openai_relevance_{retrieved_document_index}"]
            .map(lambda x: int(x == "relevant"))
            .to_numpy()
        )

        query_and_evaluations_df[f"openai_precision@{num_retrieved_documents}"] = pd.Series(
            num_relevant_documents_array / num_retrieved_documents
        )

    return query_and_evaluations_df
