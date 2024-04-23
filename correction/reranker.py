import abc
from typing import Optional, Any, Union

import numpy as np

from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.util import cos_sim

from correction.utils import template


class ReRanker(metaclass=abc.ABCMeta):
    def __init__(
        self,
        model: Optional[Union[SentenceTransformer, CrossEncoder]] = None,
        model_name: str = "",
        max_tokens: int = 300,
        top_k: int = 10
    ):
        if not model_name and not model:
            raise ValueError("No model provided, need at least 'model' or 'model_name'")
        self.model_name = model_name
        self.model = None
        if model:
            self.model = model
        self.max_tokens = max_tokens
        self.top_k = top_k

    @staticmethod
    def _resort_output(positions: list[int], distances: list[float], **kwargs):
        output = {k: [[v[0][p] for p in positions]] for k, v in kwargs.items()}
        output.update({"distances": [distances]})
        return output

    @abc.abstractclassmethod
    def _rerank(self, query: Union[str, list[list[float]]], documents: list[str], metadatas: list[dict[str, Any]]):
        raise NotImplementedError

    def __call__(self, **kwargs):
        rerank_params = {k: kwargs.get(k) for k in ["query", "documents", "metadatas"]}
        positions, distances = self._rerank(**rerank_params)
        out_params = {k: kwargs.get(k) for k in ["ids", "documents", "metadatas"]}
        out_params.update(dict(positions=positions, distances=distances))
        return self._resort_output(**out_params)


class CrossEncoderReRanker(ReRanker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.model and self.model_name:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_tokens,
                device="cpu"
            )

    def _rerank(
            self,
            query: Union[str, list[list[float]]],
            documents: list[str],
            metadatas: list[dict[str, Any]]
    ):
        titles = [m["title"] for m in metadatas[0]]
        templates = template(titles, documents[0])
        texts = templates["doc"]

        if not isinstance(query, str):
            raise ValueError("The query must be a str for cross-encoder reranking.")

        result = self.model.rank(query, texts, top_k=self.top_k)
        result = result[::-1] # most relevant -> last.
        positions = [r["corpus_id"] for r in result]
        scores = [float(r["score"]) for r in result]

        return positions, scores


class MRLEmbeddingReRanker(ReRanker):
    def __init__(self, dimensions: int = 768, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = dimensions
        if not self.model and self.model_name:
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.model.max_seq_length = self.max_tokens

    def _embed(self, examples: list[str], verbose: bool = False) -> list[list[float]]:
        embeddings = self.model.encode(examples, show_progress_bar=verbose, convert_to_numpy=True)
        return embeddings[:, :self.dimensions].tolist()

    def _rerank(
            self,
            query: Union[str, list[list[float]]],
            documents: list[str],
            metadatas: list[dict[str, Any]]
    ):
        titles = [m["title"] for m in metadatas[0]]
        templates = template(titles, documents[0])

        if isinstance(query, str):
            q_emb = self._embed([query])
        else:
            q_emb = query

        doc_embs = self._embed(templates["doc"])

        cosines = cos_sim(doc_embs, q_emb).numpy()
        cosines = cosines[:, 0].tolist()
        positions = np.argsort(cosines, axis=0).tolist()
        positions = positions[-self.top_k:]

        return positions, cosines