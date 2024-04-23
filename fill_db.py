import argparse

from tqdm import tqdm
from datasets import load_from_disk

from correction.db import ChromaDB
from correction.definitions import ClinicalNotes, ActionResults
from correction.logging import prepare_log

note = """first-line treatment for Eclamptic"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-fs", "--few_shot_file", type=str, default="")
    parser.add_argument("-mw", "--medwiki_path", type=str, default="")
    parser.add_argument("-em", "--embedding_model_path", type=str, default="./pubmedbert-base-embeddings")
    parser.add_argument("-dm", "--db_metric", type=str, default="cosine")
    args = parser.parse_args()

    is_few_shot = bool(args.few_shot_file)
    assert is_few_shot ^ bool(args.medwiki_path), "Need to use only one source: few shots or medwiki."

    log = prepare_log()
    db = ChromaDB(path=args.path, log=log, model=args.embedding_model_path, db_metric=args.db_metric, rerank="none")

    if db.is_empty():
        if args.few_shot_file:
            few_shots = ClinicalNotes.from_json_path(args.few_shot_file, -1)
            db.fill(few_shots, verbose=True)
        if args.medwiki_path:
            data = load_from_disk(args.medwiki_path)

            log.info("Sort Data by Length")
            len_lambda = lambda x: {"len": [len(i) for i in x]}
            data = data.map(len_lambda, input_columns="text", batch_size=4096, batched=True, num_proc=6)
            data = data.sort("len")
            data = data.remove_columns(["len"])

            data = data.rename_column("wiki_id", "doc_id")
            data = data.add_column("source", ["medical_wikipedia"] * len(data))

            data.set_format("pandas")

            log.info("Filling DB")
            batch_size = 4096
            for d in tqdm(data.iter(batch_size), total=int(len(data)/batch_size) + 1):
                db.fill(d)

    # Test on note
    out: ActionResults = db.fetch(note, 20, is_few_shot=is_few_shot)
    db.log.info(out)

    if args.path:
        # Dump if actual path
        db.dump()
