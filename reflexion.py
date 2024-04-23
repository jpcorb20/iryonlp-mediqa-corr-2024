# encoding: utf-8
import argparse
import json
import asyncio
import logging

import pandas as pd

from correction.engine import SKEngine
from correction.db import ChromaDB
from correction.logging import prepare_log

from react import ReAct


class Memory:
    def __init__(self):
        self.reflexions = []

    def append(self, text: str):
        self.reflexions.append(text)

    def __str__(self):
        delimiter = "\n------------------------\n"
        return delimiter.join(f"REFLEXION {i+1}\n{m}" for i, m in enumerate(self.reflexions))


class ReFlex:
    def __init__(
        self,
        engine: SKEngine,
        db: ChromaDB,
        log: logging.Logger,
        output_filename: str = "output",
        turns: int = 5,
        react_turns: int = 4,
        instructions: str = "",
        review_thresholds: tuple[float, float] = (3.8, 3.0)
    ):
        self.engine = engine
        self.db = db
        self.log = log
        self.turns = turns
        self.output_filename = output_filename
        self.instructions = instructions
        self.react = ReAct(engine, db, log, turns=react_turns, instructions=instructions) # , keep_only_last_results=True
        self.last_output = None
        self.tracking = {}
        self.review_thresholds = review_thresholds

    def reflex(self, text: str, sentences: str):
        long_term_memory = Memory()
        done = False
        engine = self.engine
        log = self.log
        max_count = self.turns
        count = 0
        parsed_outputs = None
        review_thresholds = self.review_thresholds
        self.tracking = {"reflexion_count": 0, "react_counts": [], "react_final_answers": [], "reviews": [], "runtime": [], "sources": []}
        while count < max_count and not done:
            log.info("REFLEXION STEP %d" % count)

            memory = str(long_term_memory)

            history, react_count, is_final_flag, avg_time, sources, reviews = self.react(text, memory, parse_output=False, evaluate=True)
            log.info(history)

            if is_final_flag:
                parsed_outputs = asyncio.run(
                    engine.find_and_correct(clinical_note=sentences, search_history=str(history), memory=memory)
                )

            if reviews:
                meta_eval, min_eval = reviews.meta_evaluation()
                log.info(str(meta_eval))
                log.info(str(min_eval))

            self.tracking["react_counts"].append(react_count)
            self.tracking["react_final_answers"].append(is_final_flag)
            self.tracking["runtime"].append(avg_time)
            self.tracking["sources"].append(sources)
            self.tracking["reviews"].append({"avg": meta_eval.final_score, "min": min_eval.final_score} if reviews else None)

            if not is_final_flag or (reviews and (meta_eval.final_score < review_thresholds[0] or min_eval.final_score < review_thresholds[1])):
                engine.refresh_token_and_reload_service()
                reflexion = asyncio.run(
                    engine.reflexion(clinical_note=text, search_history=str(history), reviews=str(reviews), memory=memory, instructions=self.instructions)
                )
                log.info(reflexion[0])
                long_term_memory.append(reflexion[0])
                count += 1
                self.tracking["reflexion_count"] = count
            else:
                done = True
                break

        if parsed_outputs is None:
            parsed_outputs = asyncio.run(
                engine.find_and_correct(clinical_note=sentences, search_history=str(history), memory=memory)
            )

        self.last_output = parsed_outputs[0]
        log.info(self.last_output)

        return self.last_output

    def __call__(self, *args, **kwargs):
        return self.reflex(*args, **kwargs)

    def export_tracking(self, id: str):
        local_obj = self.tracking
        local_obj.update({"id": id})
        output = json.dumps(self.tracking)
        with open(f"{self.output_filename}.jsonl", "a") as fp:
            fp.write(output + "\n")

    def export_last_output(self, id: str):
        assert self.last_output is not None, "last_output must be set by running .reflex()."
        res = self.last_output.to_tuple(id)
        with open(f"{self.output_filename}.txt", "a") as fp:
            fp.write('%s %d %d "%s"\n' % res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-db", "--database_path", type=str, required=True)
    parser.add_argument("-o", "--output_filename", type=str, default="output")
    parser.add_argument("-em", "--embedding_model_path", type=str, default="./embeddings/pubmedbert-base-embeddings")
    parser.add_argument("-s", "--n_samples", type=int, default=1e7)
    parser.add_argument("-sr", "--skip_rows", type=int, default=0)
    parser.add_argument("-si", "--start_index", type=int, default=0)
    parser.add_argument("-ei", "--end_index", type=int, default=100_000) # some bigger number than the expected data
    parser.add_argument("-i", "--indexes", type=str, default="")
    parser.add_argument("-r", "--reverse", action="store_true")
    parser.add_argument("-rt", "--retrieval_topk", type=int, default=50)
    parser.add_argument("-rm", "--rerank_model", type=str, default="crossencoder")
    parser.add_argument("-t", "--rerank_topk", type=int, default=20)
    parser.add_argument("-ds", "--db_source", type=str, default=None)
    parser.add_argument("-rat", "--review_avg_thresold", type=float, default=3.8)
    parser.add_argument("-rmt", "--review_min_thresold", type=float, default=3.0)
    parser.add_argument("-rtn", "--react_turns", type=float, default=4.0)
    parser.add_argument("-rtm", "--reflexion_turns", type=float, default=5.0)
    args = parser.parse_args()

    assert args.n_samples == 1e7 or not args.indexes, "Both 'n_samples' and 'indexes', cannot be used together."
    assert not args.indexes or args.skip_rows == 0, "Both 'indexes' and 'skip_rows', cannot be used together."

    log = prepare_log(level=logging.INFO)

    log.info("STARTING ENGINE")
    engine = SKEngine(log=log, skill_folder="reflexion")

    log.info("LOADING CSV")
    df = pd.read_csv(args.file, index_col=0 if "test" in args.file.lower() else 1) # load with ids as indexes.

    # Sample for fast dev, or skip values or gather indexes to continue missed inference(s).
    if not args.indexes:
        if args.n_samples < len(df):
            df = df.sample(args.n_samples, random_state=42) # fix subset of samples.
        if args.skip_rows > 0:
            df = df.iloc[args.skip_rows:, :]
        if args.reverse:
            df = df.iloc[::-1]
    else:
        df = df.loc[args.indexes.split(","), :]

    if args.start_index > 0 or args.end_index < 100_000:
        df = df.iloc[args.start_index:args.end_index, :]

    with open("medical_instructions.txt", "r") as fp:
        instructions = fp.read()

    print(df)

    db = ChromaDB(
        path=args.database_path,
        log=log,
        model=args.embedding_model_path,
        top_k=args.retrieval_topk,
        rerank_topk=args.rerank_topk,
        rerank=args.rerank_model,
        fix_source=args.db_source
    )

    review_thresholds = (args.review_avg_thresold, args.review_min_thresold)
    reflex = ReFlex(
        engine,
        db,
        log,
        output_filename=args.output_filename,
        react_turns=args.react_turns,
        turns=args.reflexion_turns,
        instructions=instructions,
        review_thresholds=review_thresholds
    )

    log.info("INFERENCE")
    for id, d in df.iterrows():
        log.info(str(id))
        reflex(d["Text"], d["Sentences"])
        reflex.export_last_output(str(id))
        reflex.export_tracking(str(id))
