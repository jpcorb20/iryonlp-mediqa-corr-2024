# encoding: utf-8
import os
import argparse
import asyncio
import logging

import pandas as pd
from tenacity import RetryError

from correction.engine import SKEngine
from correction.definitions import ClinicalNotes
from correction.logging import prepare_log
from correction.db import FewShotDB


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-fs", "--few_shot_file", type=str, default="")
    parser.add_argument("-n", "--n_few_shots", type=int, default=10)
    parser.add_argument("-s", "--n_samples", type=int, default=1e7)
    parser.add_argument("-sr", "--skip_rows", type=int, default=0)
    parser.add_argument("-i", "--indexes", type=str, default="")
    parser.add_argument("-o", "--output_filename", type=str, default="output")
    args = parser.parse_args()

    assert args.n_samples == 1e7 or not args.indexes, "Both 'n_samples' and 'indexes', cannot be used together."
    assert not args.indexes or args.skip_rows == 0, "Both 'indexes' and 'skip_rows', cannot be used together."

    log = prepare_log(level=logging.DEBUG) #level=logging.INFO

    log.info("STARTING ENGINE")
    engine = SKEngine(log=log)

    if args.few_shot_file and args.n_few_shots > 0:
        base_name = os.path.basename(args.few_shot_file)
        if base_name.startswith("chromadb"):
            log.info("LOAD FEW SHOTS IN CHROMADB")
            few_shot_gen = FewShotDB(args.few_shot_file, log=log)  # model="./pubmedbert-base-embeddings"
        else:
            log.info("LOAD FEW SHOTS IN MEMORY")
            few_shot_gen = ClinicalNotes.from_json_path(args.few_shot_file, args.n_few_shots)
    else:
        few_shot_gen = None

    log.info("LOADING CSV")
    df = pd.read_csv(args.file, index_col=1) # load with ids as indexes.

    # Sample for fast dev, or skip values or gather indexes to continue missed inference(s).
    if not args.indexes:
        if args.n_samples < len(df):
            df = df.sample(args.n_samples, random_state=42) # fix subset of samples.
        if args.skip_rows > 0:
            df = df.iloc[args.skip_rows:, :]
    else:
        df = df.loc[args.indexes.split(","), :]

    log.info("INFERENCE")
    not_done = []
    for id, d in df.iterrows():
        few_shots = ""
        if isinstance(few_shot_gen, ClinicalNotes):
            few_shot_gen.resample()
            few_shots = str(few_shot_gen)
        elif isinstance(few_shot_gen, FewShotDB):
            few_shots = str(few_shot_gen.fetch(d["Sentences"], args.n_few_shots))

        try:
            out = asyncio.run(
                engine.find_and_correct(clinical_note=d["Sentences"], few_shots=few_shots)
            )
            for o in out:
                log.info(str(o))
                if o:
                    tup = o.to_tuple(str(id))
                    with open(f"{args.output_filename}.txt", "a") as fp:
                        fp.write('%s %d %d "%s"\n' % tup)
        except RetryError:
            log.debug(f"DONE WITH MAX RETRY : {str(id)}.")
            not_done.append(str(id))
    log.info("NOT DONE: %s" % ",".join(not_done))
