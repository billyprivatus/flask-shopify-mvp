import pandas as pd

from typing import Dict, List, Optional, Tuple
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone


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
