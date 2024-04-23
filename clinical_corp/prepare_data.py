from tqdm import tqdm
from datasets import load_dataset, Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter


def parse_id(text: str):
    return int(text.split("_")[-1])


def parse_doc_id(text: str):
    return "_".join(text.split("_")[:-1])


def parse_medrag_dataset(dataset: str):
    data = load_dataset(f"./{dataset}", split="train")
    data = data.rename_column("content", "text")
    data = data.remove_columns("contents")
    params = dict(batched=True, batch_size=1_000, input_columns=["id"])
    data = data.map(lambda x: {"paragraph_id": [parse_id(i) for i in x], "doc_id": [parse_doc_id(i) for i in x]}, **params)
    data = data.remove_columns("id")
    data = data.add_column("source", [dataset] * len(data))
    data.save_to_disk(f"parsed_{dataset}")
    return data


def parse_guidelines():
    global ACC2NAMES, text_splitter
    datafiles = ["./guidelines/open_guidelines.jsonl", "./guidelines/closed_guidelines.jsonl"]
    data = load_dataset("json", data_files=datafiles, split="train")
    data = data.rename_column("clean_text", "text")
    data = data.rename_column("source", "source_short")
    data = data.rename_column("id", "doc_id")
    data = data.map(lambda x: {"source": [ACC2NAMES.get(i, "unknown") for i in x]}, input_columns=["source_short"], batched=True)
    data = data.remove_columns(["raw_text", "url", "overview", "source_short"])
    data = data.map(lambda x: {"chunks": [text_splitter.split_text(i) for i in x]}, input_columns=["text"], batched=True, num_proc=8)
    new_data = []
    for d in tqdm(data.iter(batch_size=1), total=len(data)):
        for i, c in enumerate(d["chunks"][0]):
            new_d = {k: v[0] for k, v in d.items() if k != "text" and k != "chunks"}
            new_d.update({"text": c, "paragraph_id": i})
            new_data.append(new_d)
    new_dataset = Dataset.from_list(new_data)
    new_dataset.save_to_disk("parsed_guidelines")
    return new_dataset


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

ACC2NAMES = dict(
    aafp="American Academy of Family Physicians",
    cco="Cancer Care Ontario",
    cdc="Center for Disease Control and Prevention",
    cma="Canadian Medical Association",
    cps="Canadian Paediatric Society",
    drugs="Drugs.com",
    gc="GuidelineCentral",
    idsa="Infectious Diseases Society of America",
    nice="National Institute for Health and Care Excellence",
    pubmed="PubMed",
    spor="Strategy for Patient-Oriented Research",
    who="World Health Organization",
    wikidoc="WikiDoc"
)

if __name__ == "__main__":
    # Parse data from the MedCorp
    data = parse_medrag_dataset("textbooks")

    # Parse data from guidelines
    # parse_guidelines()
