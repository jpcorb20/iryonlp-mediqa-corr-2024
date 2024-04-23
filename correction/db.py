# Workaroung to work with newest version of sqlite3 for chromadb.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import shutil
import tempfile
from uuid import uuid4
from typing import Optional, Any, Union
from logging import Logger

import pandas as pd
from pandas import DataFrame
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

from chromadb import PersistentClient, HttpClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from correction.definitions import ClinicalNotes, ActionResults
from correction.reranker import MRLEmbeddingReRanker, CrossEncoderReRanker
from correction.utils import template


class ChromaDB:
    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: str = "mediqa_corr_fs",
        model: str = "./all-MiniLM-L6-v2",
        log: Optional[Logger] = None,
        max_tokens: int = 200,
        dimensions: int = 256,
        top_k: int = 300,
        rerank: str = "mrl",
        rerank_dimensions: int = 768,
        rerank_topk: int = 20,
        rerank_cross_encoder_model_name: str = 'embeddings/MedCPT-Cross-Encoder',
        db_metric: str = "cosine",
        fix_source: str = None
    ):
        self.path = path
        self.collection_name = collection_name
        self.model = model
        self.dimensions = dimensions
        self.top_k = top_k
        self.fix_source = fix_source
        self.log = log

        if path:
            self.temp_path = ""
            if not os.path.exists(self.path):
                self.temp_path = tempfile.mkdtemp()
            if self.log:
                log.info("Setting up persistent client on temp file: %s." % self.temp_path)
            self.client = PersistentClient(path=self.temp_path or self.path)
        else:
            log.info("Setting up http client on http://localhost:8000.")
            self.temp_path = None
            self.client = HttpClient(settings=Settings(anonymized_telemetry=False))

        if self.log:
            log.info("Getting or creating collection '%s'." % collection_name)
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": db_metric})
        if self.log:
            log.info("Collection has %d records." % self.collection.count())

        if self.log:
            log.info("Loading embedding model: %s." % model)

        self.model = SentenceTransformer(model, device="cpu")
        self.model.max_seq_length = max_tokens

        self.reranker = None
        if rerank == "mrl":
            log.info("Loading MRL ReRanker")
            self.reranker = MRLEmbeddingReRanker(model=self.model, dimensions=rerank_dimensions, top_k=rerank_topk)
        elif rerank == "crossencoder":
            log.info("Loading CrossEncoder ReRanker")
            self.reranker = CrossEncoderReRanker(model_name=rerank_cross_encoder_model_name, top_k=rerank_topk)
        else:
            log.info("ReRanker turned off.")

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def _embed(self, examples: list[str], verbose: bool = False, dimensions: int = -1) -> list[list[float]]:
        embeddings = self.model.encode(examples, show_progress_bar=verbose, convert_to_numpy=True)
        dim = self.dimensions if dimensions == -1 else dimensions
        return embeddings[:, :dim].tolist()

    def _prepare_clinical_notes(self, notes: ClinicalNotes, verbose: bool = False):
        examples = notes.examples
        ids = [e.id for e in examples]
        assert all(i is not None for i in ids), "Cannot fill DB: at least one None in ids."

        metas = [e.mistake.to_dict() for e in examples]
        assert all(m is not None for M in metas for m in list(M.values())), "Cannot fill DB: at least one None in metadatas."
        if self.log:
            self.log.info("Generating embeddings for %s documents." % len(notes))
        docs = [e.text for e in examples]
        embeddings = self._embed(docs, verbose=verbose)
        return ids, docs, metas, embeddings

    def _prepare_pandas(self, notes: DataFrame, verbose):
        notes: Dataset = Dataset.from_pandas(notes)
        ids = [str(uuid4()) for _ in range(len(notes))]
        notes = notes.map(lambda x: template(x["title"], x["text"]), batch_size=128, batched=True)
        docs = notes["doc"]
        if self.log:
            self.log.info("Generating embeddings for %s documents." % len(notes))
        embeddings = self._embed(docs, verbose=verbose)
        docs = notes["text"]
        if 'wiki_id' in notes.features:
            notes = notes.remove_columns(['topic_infer', 'prob'])
        metas = notes.remove_columns(["doc", "text"])
        metas = metas.to_pandas()
        metas[pd.isna(metas)] = "None"
        metas = metas.to_dict(orient="records")
        return ids, docs, metas, embeddings

    def fill(self, notes: Union[ClinicalNotes, DataFrame], verbose: bool = False):
        if isinstance(notes, ClinicalNotes):
            ids, docs, metas, embeddings = self._prepare_clinical_notes(notes, verbose)
        elif isinstance(notes, DataFrame):
            ids, docs, metas, embeddings = self._prepare_pandas(notes, verbose)
        else:
            raise ValueError("'notes' must be ClinicalNotes or DataFrame.")

        if self.log:
            self.log.info("Filling collection with %d records." % len(notes))

        self.collection.add(
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
            ids=ids,
        )

    def _format_fs_output(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]],
        distances: list[float],
        threshold: float = 0.0,
        **kwargs
    ) -> ClinicalNotes:
        notes = []
        for i, d, m, s in zip(ids[0], documents[0], metadatas[0], distances[0]):
            new_dict = dict(**m)
            new_dict.update({"Text Id": i, "Sentences": d, "Score": s})
            notes.append(new_dict)

        notes = sorted(notes, key=lambda x: x["Score"])
        if threshold > 0.0:
            notes = list(filter(lambda x: x["Score"] >= threshold, notes))

        for n in notes:
            n.pop("Score")
        return ClinicalNotes.from_dict(notes)

    def _format_output(
        self,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        distances: list[float],
        **kwargs
    ) -> ActionResults:
        notes = []
        for m, d, s in zip(metadatas[0], documents[0], distances[0]):
            new_dict = dict(
                relevance_score=s,
                source=m.get("source", "unknown"),
                title=m.get("title"),
                paragraph_id=m.get("paragraph_id", 0),
                text=d
            )
            notes.append(new_dict)

        notes = sorted(notes, key=lambda x: x["relevance_score"])

        return ActionResults.from_list_dict(notes)

    def fetch(
        self,
        note: str,
        threshold: float = 0.0,
        is_few_shot: bool = True
    ) -> Union[ClinicalNotes, ActionResults]:
        assert not self.is_empty(), "Collection seems empty, make sure to .fill()."
        embeddings = self._embed([note])
        _include = ["documents", "metadatas", "distances"]
        search_params = {}
        source = self.fix_source
        if source:
            if source in ["medical_wikipedia", "startpearls", "textbooks"]:
                search_params["where"] = {"source": {"$eq": source}}
            elif source == "guidelines":
                names = ['American Academy of Family Physicians', 'Cancer Care Ontario', 'Center for Disease Control and Prevention', 'Canadian Medical Association', 'Canadian Paediatric Society', 'Drugs.com', 'GuidelineCentral', 'Infectious Diseases Society of America', 'National Institute for Health and Care Excellence', 'PubMed', 'Strategy for Patient-Oriented Research', 'World Health Organization', 'WikiDoc']
                guidelines = [{"source": {"$eq": t}} for t in names]
                search_params["where"] = {"$or": guidelines}
        output = self.collection.query(query_embeddings=embeddings, n_results=self.top_k, include=_include, **search_params)
        if self.reranker:
            output = self.reranker(query=note, **output)
        format_out = self._format_fs_output if is_few_shot else self._format_output
        return format_out(threshold=threshold, **output)

    def dump(self):
        if self.temp_path:
            if self.log:
                self.log.info("Dumping db from tempfile '%s' to path '%s'." % (self.temp_path, self.path))
            shutil.move(self.temp_path, self.path)
        else:
            if self.log:
                self.log.warning("Path existed. No need to dump from tempfile.")
