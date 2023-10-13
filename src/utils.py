import polars as pl

def read_csv(file):
    df = pl.read_csv(file)
    return list(df.get_columns()[0])

def read_txt(file):
    with open(file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text