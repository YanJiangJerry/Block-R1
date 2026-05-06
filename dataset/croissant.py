from mlcroissant import Dataset

ds = Dataset(jsonld="https://huggingface.co/api/datasets/YanJiangJerry/Block-R1/croissant")
records = ds.records("default")