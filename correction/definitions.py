import os
import json
import re
from typing import List, Optional, Union
from dataclasses import dataclass
from collections import Counter

import pandas as pd


@dataclass
class Mistake:
    is_error: Optional[bool] = None
    reasoning: Optional[str] = None
    sentence_id: Optional[int] = None
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None

    @staticmethod
    def na_check(x: Union[str, None]) -> str:
        return x if x else "NA"

    def __str__(self):
        skip_fields = ["reasoning"]  # ["sentence_id", "sentences"]
        na_check = self.na_check
        identity = lambda x: x
        type_map = {"is_error": int, "corrected_sentence": na_check, "error_sentence": na_check}
        obj = {k: type_map.get(k, identity)(v) for k, v in self.__dict__.items() if k not in skip_fields}
        return json.dumps(obj, indent=4, ensure_ascii=False)

    def to_tuple(self, id: str = "") -> tuple[str, int, int, str]:
        return (
            str(id),
            int(self.is_error),
            self.sentence_id if self.sentence_id else -1,
            self.na_check(self.corrected_sentence)
        )

    def to_dict(self) -> dict:
        return {
            "Error Flag": int(self.is_error),
            "Reasoning": self.reasoning,
            "Error Sentence ID": self.sentence_id if self.sentence_id else -1,
            "Error Sentence": self.na_check(self.error_sentence),
            "Corrected Sentence": self.na_check(self.corrected_sentence)
        }

    @staticmethod
    def parse_sentences(sentences: str) -> list[str]:
        output = re.split(r"\n\d+\s", sentences)
        # Remove number with space in first sentence since not in pattern.
        output[0] = output[0][2:]
        return output

    @classmethod
    def from_str(cls, text: str) -> "Mistake":
        try:
            object = json.loads(text)
            return cls(**object)
        except json.decoder.JSONDecodeError:
            print("Could not parse string into Mistake.")

    @classmethod
    def from_json(cls, object: dict) -> "Mistake":
        if "sentences" in object and object["sentences"]:
            object["sentences"] = cls.parse_sentences(object["sentences"])
        return cls(**object)


@dataclass
class ClinicalNote:
    text: str
    mistake: Optional[Mistake] = None
    id: Optional[str] = None

    @classmethod
    def from_dict(cls, obj: dict) -> "ClinicalNote":
        is_error = bool(obj.get("Error Flag"))
        mistake = dict(
            is_error=is_error,
            reasoning=obj.get("Reasoning"),
            sentence_id=obj.get("Error Sentence ID"),
            error_sentence=obj.get("Error Sentence") if is_error else None,
            corrected_sentence=obj.get("Corrected Sentence") if is_error else None
        )
        new_obj = dict(
            id=obj.get("Text Id"),
            text=obj.get("Sentences"),
            mistake=Mistake.from_json(mistake)
        )
        return cls(**new_obj)

    def __str__(self) -> str:
        output = f"CLINICAL NOTE:\n{self.text}\n"
        if self.mistake:
            output += f"OUTPUT:{str(self.mistake)}"
        return output


@dataclass
class ClinicalNotes:
    examples: Optional[List[Union[ClinicalNote, dict]]] = None
    _raw_df: Optional[pd.DataFrame] = None
    _n_shot: Optional[int] = None
    _only_errors: Optional[bool] = None

    def resample(self):
        raw_df = self._raw_df
        if not self._only_errors and self._n_shot % 2 == 0:
            n = int(self._n_shot/2)
            errors = raw_df["Error Flag"]
            error_df = raw_df[errors == 1].sample(n=n)
            noerror_df = raw_df[errors == 0].sample(n=n)
            df = pd.concat([error_df, noerror_df])
            df = df.sample(frac=1)
        elif self._n_shot > 0:
            df = raw_df.sample(n=self._n_shot)
        else:
            df = raw_df
        records = df.reset_index(names="Text Id").to_dict(orient='records')
        self.examples = [ClinicalNote.from_dict(i) for i in records]

    @classmethod
    def from_dict(cls, records: list[dict]):
        return cls([ClinicalNote.from_dict(i) for i in records])

    @classmethod
    def from_json_path(cls, path: str, n_shot: int = 1, only_errors: bool = False):
        df = pd.read_csv(path, index_col=1)
        df.drop("Unnamed: 0", axis=1, inplace=True)

        base_path = "/".join(path.split("/")[:-1])
        reasons = pd.read_csv(os.path.join(base_path, "train_reasoning.tsv"), sep="\t", index_col=0)
        df = df.join(reasons)

        if only_errors:
            df = df[df["Error Flag"] == 1]

        obj = cls(_raw_df=df, _n_shot=n_shot, _only_errors=only_errors)
        obj.resample()
        return obj

    def __len__(self):
        return len(self.examples)

    def __str__(self):
        return "\n\n".join([str(i) for i in self.examples])


@dataclass
class Action:
    function: str
    parameters: list

    @classmethod
    def from_str(cls, text: str) -> "Action":
        elems = text.split("(")
        function = elems[0]
        params = "(".join(elems[1:])[:-1].split('", "')
        parameters = [p.replace('"', '') for p in params]
        return cls(function=function, parameters=parameters)

    def __str__(self):
        parameters = '"' + '", "'.join(self.parameters) + '"'
        return f"{self.function}({parameters})"


@dataclass
class ActionResult:
    title: str
    text: str
    source: Optional[str] = None
    paragraph_id: Optional[int] = None
    relevance_score: Optional[float] = None

    def to_dict(self):
        _keys = ("relevance_score", "source", "title", "paragraph_id", "text")
        output = {}
        for k in _keys:
            v= getattr(self, k)
            if v:
                output[k] = v
        return output

    def __str__(self):
        output = json.dumps(self.to_dict(), indent=4, ensure_ascii=False)
        return output.replace("\n", "\n    ")


@dataclass
class ActionResults:
    results: list[ActionResult]

    @classmethod
    def from_list_dict(self, objs: list[dict]) -> "ActionResults":
        results = [ActionResult(**x) for x in objs]
        return ActionResults(results)

    def to_dict(self):
        return [r.to_dict() for r in self.results]

    def get_sources(self) -> list[str]:
        return [r.source for r in self.results]

    def count_sources(self) -> dict:
        counts = Counter(self.get_sources())
        return dict(counts)

    def __str__(self):
        results = [str(r) for r in self.results]
        return "[\n    " + ",\n    ".join(results) + "\n]"


@dataclass
class ReACTStep:
    observation: str
    thought: str
    action: Action
    action_results: Optional[ActionResults] = None

    @classmethod
    def from_str(cls, text: str) -> "ReACTStep":
        try:
            obj = json.loads(text)
            obj["action"] = Action.from_str(obj["action"])
            return cls(**obj)
        except json.JSONDecodeError:
            raise ValueError("Cannot parse the JSON into a ReACTStep.")

    def set_action_results(self, results: Optional[ActionResults] = None):
        self.action_results = results

    def __str__(self):
        obj = {k: (v.to_dict() if isinstance(v, ActionResults) else str(v)) for k, v in self.__dict__.items() if v is not None}
        return json.dumps(obj, indent=4, ensure_ascii=False)


@dataclass
class ReACTHistory:
    steps: Optional[list[ReACTStep]] = None
    keep_only_last_results: bool = True

    def __post_init__(self):
        if not self.steps:
            self.steps = []

    def reset_action_results(self):
        [s.set_action_results() for s in self.steps]

    def append(self, element: ReACTStep):
        if self.keep_only_last_results and len(self.steps) > 0:
            self.reset_action_results()
        self.steps.append(element)

    def __str__(self):
        lines = [str(s).replace("\n", "\n    ").replace("\\\\", "\\") for s in self.steps]
        return "[\n    " + ",\n    ".join(lines) + "\n]"


@dataclass
class Evaluation:
    validity_reasoning: str
    validity_score: float
    preciseness_reasoning: str
    preciseness_score: float
    confidence_reasoning: str
    confidence_score: float
    relevance_reasoning: str
    relevance_score: float
    completeness_reasoning: str
    completeness_score: float
    final_reasoning: str
    final_score: float

    @classmethod
    def from_str(cls, text: str) -> "Evaluation":
        try:
            object = json.loads(text)
            keywords = ["validity", "preciseness", "confidence", "relevance", "completeness", "final"]
            object = {k: v for k, v in object.items() if any(k.startswith(i) for i in keywords)}
            return cls(**object)
        except json.decoder.JSONDecodeError:
            print("Could not parse string into Evaluation.")

    def to_dict(self, reduce: bool = False):
        obj = self.__dict__
        if reduce:
            keys = ("final_reasoning", "validity_score", "preciseness_score", "confidence_score", "relevance_score", "completeness_score", "final_score")
            obj = {k: v for k, v in obj.items() if k in keys}
        return obj

    def to_str(self, reduce: bool = False):
        obj = self.to_dict(reduce=reduce)
        return json.dumps(obj, indent=4, ensure_ascii=False)

    def __str__(self):
        return self.to_str()

@dataclass
class MetaEvaluator:
    evaluations: list[Evaluation]

    @classmethod
    def from_list_str(cls, texts: list[str]) -> "MetaEvaluator":
        evals = [Evaluation.from_str(t) for t in texts]
        evals = [e for e in evals if e]
        return cls(evals)

    def meta_evaluation(self) -> tuple[Evaluation, Evaluation]:
        output = {}
        min_output = {}
        for i, e in enumerate(self.evaluations):
            for k, v in e.to_dict().items():
                if k.endswith("_score"):
                    if isinstance(v, float) or isinstance(v, int):
                        if k in output:
                            output[k] = (i * output[k] + v) / (i+1)
                            min_output[k] = min(output[k], v)
                        else:
                            output[k] = v
                            min_output[k] = v
                elif k.endswith("_reasoning") and i == 0:
                    output[k] = ""
                    min_output[k] = ""
        return Evaluation(**output), Evaluation(**min_output)

    def __str__(self):
        evals = [e.to_dict(reduce=True) for e in self.evaluations]
        return json.dumps(evals, indent=4, ensure_ascii=False)
