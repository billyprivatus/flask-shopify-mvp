import numpy as np
import phoenix as px


def str_to_number_array(str):
    numbers_str = str.strip('[]').split()
    numbers_str_list = numbers_str[0].split(',')
    data_array = np.array([float(num) for num in numbers_str_list])
    return data_array


def data_prep(query_df, database_df):
    query_df['text_vector'] = query_df['text_vector'].apply(
        str_to_number_array)
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

    [query_df, database_df] = data_prep(query_df, database_df)

    query_schema = px.Schema(
        prompt_column_names=px.EmbeddingColumnNames(
            raw_data_column_name="text",
            vector_column_name="centered_text_vector",
        ),
        response_column_names="response",
        tag_column_names=[
            "context_text_0",
            "context_similarity_0",
            "context_text_1",
            "context_similarity_1",
            # "euclidean_distance_0",
            # "euclidean_distance_1",
            "openai_relevance_0",
            "openai_relevance_1",
            "openai_precision@1",
            "openai_precision@2",
        ],
    )

    database_schema = px.Schema(
        document_column_names=px.EmbeddingColumnNames(
            raw_data_column_name="text",
            vector_column_name="centered_text_vector",
        ),
    )

    # Dataset
    prim_ds = px.Dataset(
        dataframe=query_df,
        schema=query_schema,

        name="query",
    )
    corpus_ds = px.Dataset(
        dataframe=database_df,
        schema=database_schema,
        name="pinecone",
    )

    session = px.launch_app(
        primary=prim_ds, corpus=corpus_ds, port=6060)

    return session
