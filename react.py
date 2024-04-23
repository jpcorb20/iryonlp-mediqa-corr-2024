# encoding: utf-8
import argparse
import asyncio
import logging
import time

import pandas as pd

from correction.engine import SKEngine
from correction.db import ChromaDB
from correction.logging import prepare_log
from correction.definitions import ReACTHistory, MetaEvaluator


class ReAct:
    def __init__(
        self,
        engine: SKEngine,
        db: ChromaDB,
        log: logging.Logger,
        output_filename: str = "output",
        turns: int = 20,
        keep_only_last_results: bool = False,
        instructions: str = ""
    ):
        self.engine = engine
        self.db = db
        self.log = log
        self.output_filename = output_filename
        self.turns = turns
        self.keep_only_last_results = keep_only_last_results
        self.last_output = None
        self.instructions = instructions

    def react(self, text: str, memory: str = "", sentences: str = "", parse_output: bool = True, evaluate: bool = False):
        history = ReACTHistory(keep_only_last_results=self.keep_only_last_results)
        done = False
        count = 0
        engine = self.engine
        log = self.log
        max_count = self.turns
        meta_eval = None
        avg_time = 0
        sources = []
        mem_param = {}
        if memory:
            mem_param["memory"] = memory
        while count < max_count:
            start_time = time.time()
            # Re-run turn if asyncio error out "loop event closed".
            try:
                log.info("REACT STEP %d" % count)
                out = asyncio.run(
                    engine.react(clinical_note=text, history=str(history), n_turns=max_count, instructions=self.instructions, **mem_param)
                )
                for o in out:
                    log.info(str(o))

                if out[0]:
                    step = out[0]
                    if step.action.function == "search":
                        log.info("SEARCHING")
                        results = self.db.fetch(
                            step.action.parameters[0],
                            is_few_shot=False
                        )
                        log.info(str(results))
                        step.set_action_results(results)
                        sources.append(results.count_sources())
                        history.append(step)
                    elif step.action.function == "final_flagged_mistake":
                        log.info(step.action.parameters)
                        if evaluate:
                            engine.refresh_token_and_reload_service()
                            meta_eval: MetaEvaluator = asyncio.run(
                                engine.evaluation(clinical_note=text, search_history=str(history), **mem_param)
                            )
                            log.info(str(meta_eval))
                        history.append(step)
                        done=True
                count += 1
                if count == max_count or done:
                    break
            except RuntimeError as e:
                log.debug(e)

            delta_time = time.time() - start_time
            avg_time = ((count - 1) * avg_time + delta_time) / count

            # Hack to avoid request timeout.
            engine.refresh_token_and_reload_service()

        if parse_output:
            parsed_outputs = asyncio.run(
                engine.find_and_correct(clinical_note=sentences, search_history=str(history))
            )
            self.last_output = parsed_outputs[0]

        log.info("# Turns: %d" % count)

        output = (history, count, done, avg_time, sources)
        if evaluate:
            output += (meta_eval,)
        return output

    def __call__(self, *args, **kwargs):
        return self.react(*args, **kwargs)

    def export_last_output(self, id: str):
        assert self.last_output is not None, "last_output must be set by running .react()."
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
    parser.add_argument("-i", "--indexes", type=str, default="")
    args = parser.parse_args()

    assert args.n_samples == 1e7 or not args.indexes, "Both 'n_samples' and 'indexes', cannot be used together."
    assert not args.indexes or args.skip_rows == 0, "Both 'indexes' and 'skip_rows', cannot be used together."

    log = prepare_log(level=logging.DEBUG) #level=logging.INFO

    log.info("STARTING ENGINE")
    engine = SKEngine(log=log, skill_folder="react")

    log.info("LOADING CSV")
    df = pd.read_csv(args.file, index_col=0) # load with ids as indexes.

    # Sample for fast dev, or skip values or gather indexes to continue missed inference(s).
    if not args.indexes:
        if args.n_samples < len(df):
            df = df.sample(args.n_samples, random_state=42) # fix subset of samples.
        if args.skip_rows > 0:
            df = df.iloc[args.skip_rows:, :]
    else:
        df = df.loc[args.indexes.split(","), :]

    db = ChromaDB(path=args.database_path, log=log, model=args.embedding_model_path)
    react = ReAct(engine, db, log, args.output_filename)

    log.info("INFERENCE")
    for id, d in df.iterrows():
        history, count, done, avg_time, sources = react(d["Text"], sentences=d["Sentences"])
        react.export_last_output(str(id))
        print(history)
