import math
import pandas as pd
import numpy as np

from typing import Dict, List, Optional, Tuple
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone

# helpers


def list_vector_ids_from_query_by_namespace(index, query_vector, namespace=""):
    results = index.query(vector=query_vector, top_k=10000,
                          namespace=namespace, include_values=False)

    return list(map(lambda x: x.id, results.matches))


def list_all_vector_ids_by_namespace(index, namespace="", num_dimensions=1536, injected_stats=None):
    stats = index.describe_index_stats() if injected_stats is None else injected_stats
    namespace_map = stats['namespaces']
    vector_count = namespace_map[namespace]['vector_count']
    all_ids = set()

    while len(all_ids) < vector_count:
        query_vector = np.random.rand(num_dimensions).tolist()
        ids = list_vector_ids_from_query_by_namespace(
            index=index, query_vector=query_vector, namespace=namespace)
        all_ids.update(set(ids))

    return list(all_ids)


def get_all_vector_ids_to_vector_by_namespace(index, namespace="", num_dimensions=1536, injected_stats=None, max_batch_size=1000):
    # This function return dictionary { vector_id: vector }
    all_ids = list_all_vector_ids_by_namespace(
        index=index,
        namespace=namespace,
        num_dimensions=num_dimensions,
        injected_stats=injected_stats
    )
    batch_array_ids = np.array_split(
        all_ids, math.ceil(len(all_ids) / max_batch_size))
    result = {}

    for array_ids in batch_array_ids:
        fetch_result = index.fetch(array_ids.tolist(), namespace)
        result.update(fetch_result.vectors)

    return result


def get_pinecone_database_df_by_namespace(index, namespace="", num_dimensions=1536, injected_stats=None):
    all_vector_ids_to_vector = get_all_vector_ids_to_vector_by_namespace(
        index=index,
        namespace=namespace,
        num_dimensions=num_dimensions,
        injected_stats=injected_stats
    )
    texts = []
    text_vectors = []

    for vector in list(all_vector_ids_to_vector.values()):
        texts.append(vector.metadata["text"])
        text_vectors.append(np.array(vector.values))

    return pd.DataFrame({"text": texts, "text_vector": text_vectors})


class PineconeWrapper(Pinecone):
    query_text_to_document_score_tuples: Dict[str,
                                              List[Tuple[Document, float]]] = {}

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 4,
            filter: Optional[dict] = None,
            namespace: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        document_score_tuples = super().similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            namespace=namespace,
        )
        self.query_text_to_document_score_tuples[query] = document_score_tuples
        return document_score_tuples

    @property
    def retrieval_dataframe(self) -> pd.DataFrame:
        query_texts = []
        document_texts = []
        retrieval_ranks = []
        scores = []
        for query_text, document_score_tuples in self.query_text_to_document_score_tuples.items():
            print('query_text =', query_text)
            for retrieval_rank, (document, score) in enumerate(document_score_tuples):
                query_texts.append(query_text)
                document_texts.append(document.page_content)
                retrieval_ranks.append(retrieval_rank)
                scores.append(score)
        return pd.DataFrame.from_dict(
            {
                "query_text": query_texts,
                "document_text": document_texts,
                "retrieval_rank": retrieval_ranks,
                "score": scores,
            }
        )
