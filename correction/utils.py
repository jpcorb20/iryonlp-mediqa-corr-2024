def template(titles: list[str], texts: list[str]) -> dict[str, list[str]]:
    return {"doc": ["TITLE: %s\nTEXT: %s" % i for i in zip(titles, texts)]}
