import numpy as np
import phoenix as px


def data_prep(query_df, database_df):
    query_df['text_vector'] = query_df['text_vector'].apply(np.array)

    database_centroid = database_df["text_vector"].mean()
    database_df["centered_text_vector"] = database_df["text_vector"].apply(
        lambda x: x - database_centroid
    )

    query_centroid = query_df["text_vector"].mean()
    query_df["centered_text_vector"] = query_df["text_vector"].apply(
        lambda x: x - query_centroid)

    return [query_df, database_df]


def get_arize_url(query_df, database_df):
    px.close_app()

    # [query_df, database_df] = data_prep(query_df, database_df)

    # num_sampled_point = 500
    retrieved_document_ids = set(
        [
            doc_id
            for doc_ids in query_df["document_ids"].to_list()
            for doc_id in doc_ids
        ]
    )

    retrieved_document_mask = database_df["id"].isin(retrieved_document_ids)
    num_retrieved_documents = len(retrieved_document_ids)
    # num_additional_samples = num_sampled_point - num_retrieved_documents
    unretrieved_document_mask = ~retrieved_document_mask
    num_unretrieved_documents = unretrieved_document_mask.sum()

    sampled_unretrieved_document_ids = set(
        database_df[unretrieved_document_mask]["id"]
        .sample(n=num_unretrieved_documents, random_state=42, replace=False)
        .to_list()
    )
    sampled_unretrieved_document_mask = database_df["id"].isin(
        sampled_unretrieved_document_ids
    )
    sampled_document_mask = retrieved_document_mask | sampled_unretrieved_document_mask
    sampled_database_df = database_df[sampled_document_mask]

    database_schema = px.Schema(
        prediction_id_column_name="id",
        prompt_column_names=px.EmbeddingColumnNames(
            vector_column_name="text_vector",
            raw_data_column_name="text",
        ),
    )
    database_ds = px.Dataset(
        dataframe=sampled_database_df,
        schema=database_schema,
        name="database",
    )

    query_df['context_scores'] = query_df.apply(lambda row: np.array(
            [row['context_similarity_0'], row['context_similarity_1']]), axis=1)

    query_df = query_df.rename(columns={
        '_id': ':id.id:',
        'text': ':feature.text:prompt',
        'text_vector': ':feature.[float].embedding:prompt',
        'response': ':prediction.text:response',
        'context_similarity_0': ':tag.float:document_similarity_0',
        'context_similarity_1': ':tag.float:document_similarity_1',
        'document_ids': ':feature.[str].retrieved_document_ids:prompt',
        'context_scores': ':feature.[float].retrieved_document_scores:prompt',
        'openai_relevance_0': ':tag.str:openai_relevance_0',
        'openai_relevance_1': ':tag.str:openai_relevance_1',
        'openai_precision@1': ':tag.float:openai_precision_at_1',
        'openai_precision@2': ':tag.float:openai_precision_at_2'
    })

    query_df = query_df.drop(columns=['context_text_0', 'context_text_1'])
    query_ds = px.Dataset.from_open_inference(query_df)

    session = px.launch_app(primary=query_ds, corpus=database_ds)

    return session

    # query_schema = px.Schema(
    #     prompt_column_names=px.EmbeddingColumnNames(
    #         raw_data_column_name="text",
    #         vector_column_name="centered_text_vector",
    #     ),
    #     response_column_names="response",
    #     tag_column_names=[
    #         "context_text_0",
    #         "context_similarity_0",
    #         "context_text_1",
    #         "context_similarity_1",
    #         "euclidean_distance_0",
    #         "euclidean_distance_1",
    #         "openai_relevance_0",
    #         "openai_relevance_1",
    #         "openai_precision@1",
    #         "openai_precision@2",
    #     ],
    # )

    # database_schema = px.Schema(
    #     document_column_names=px.EmbeddingColumnNames(
    #         raw_data_column_name="text",
    #         vector_column_name="centered_text_vector",
    #     ),
    # )

    # # Dataset
    # prim_ds = px.Dataset(
    #     dataframe=query_df,
    #     schema=query_schema,

    #     name="query",
    # )
    # corpus_ds = px.Dataset(
    #     dataframe=database_df,
    #     schema=database_schema,
    #     name="pinecone",
    # )

    # session = px.launch_app(
    #     primary=prim_ds, corpus=corpus_ds, port=6060)

    # return session
