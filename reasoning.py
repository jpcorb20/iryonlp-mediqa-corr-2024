# encoding: utf-8
import argparse
import asyncio
import logging

import pandas as pd
from tenacity import RetryError

from correction.engine import SKEngine
from correction.definitions import ClinicalNote
from correction.logging import prepare_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-sr", "--skip_rows", type=int, default=0)
    parser.add_argument("-i", "--indexes", type=str, default="")
    parser.add_argument("-o", "--output_filename", type=str, default="output")
    args = parser.parse_args()

    assert not args.indexes or args.skip_rows == 0, "Both 'indexes' and 'skip_rows', cannot be used together."

    log = prepare_log(level=logging.DEBUG) #level=logging.INFO

    log.info("STARTING ENGINE")
    engine = SKEngine(log=log)

    log.info("LOADING CSV")
    df = pd.read_csv(args.file) # load with ids as indexes.

    # skip values or gather indexes to continue missed inference(s).
    if args.skip_rows > 0:
        df = df.iloc[args.skip_rows:, :]
    elif args.indexes:
        df = df.loc[args.indexes.split(","), :]

    log.info("INFERENCE")
    not_done = []
    for d in df.to_dict(orient="records"):
        id = d.get("Text ID")
        log.info("ID: " + str(id))
        log.info(d)
        cn = ClinicalNote.from_dict(d)
        try:
            out = asyncio.run(engine.correct_reasoning(clinical_note=str(cn)))
            log.info(str(out))

            if out[0]:
                with open(f"{args.output_filename}.txt", "a") as fp:
                    fp.write('%s\t"%s"\n' % (str(id), str(out[0].replace("\n", ""))))
        except RetryError:
            log.debug(f"DONE WITH MAX RETRY : {str(id)}.")
            not_done.append(str(id))
    log.info("NOT DONE: %s" % ",".join(not_done))
