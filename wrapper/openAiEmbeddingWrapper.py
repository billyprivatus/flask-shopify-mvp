import numpy as np
import pandas as pd

from typing import Dict, List, Optional
from langchain.embeddings.openai import OpenAIEmbeddings


class OpenAIEmbeddingsWrapper(OpenAIEmbeddings):
    """
    A wrapper around OpenAIEmbeddings that stores the query and document
    embeddings.
    """

    query_text_to_embedding: Dict[str, List[float]] = {}
    document_text_to_embedding: Dict[str, List[float]] = {}

    def embed_query(self, text: str) -> List[float]:
        embedding = super().embed_query(text)
        self.query_text_to_embedding[text] = embedding
        return embedding

    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = 0) -> List[List[float]]:
        embeddings = super().embed_documents(texts, chunk_size)
        for text, embedding in zip(texts, embeddings):
            self.document_text_to_embedding[text] = embedding
        return embeddings

    @property
    def query_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.query_text_to_embedding)

    @property
    def document_embedding_dataframe(self) -> pd.DataFrame:
        return self._convert_text_to_embedding_map_to_dataframe(self.document_text_to_embedding)

    @staticmethod
    def _convert_text_to_embedding_map_to_dataframe(
            text_to_embedding: Dict[str, List[float]]
    ) -> pd.DataFrame:
        texts, embeddings = map(list, zip(*text_to_embedding.items()))
        embedding_arrays = [np.array(embedding) for embedding in embeddings]
        return pd.DataFrame.from_dict(
            {
                "text": texts,
                "text_vector": embedding_arrays,
            }
        )
