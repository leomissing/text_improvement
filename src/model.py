import polars as pl
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')


def mean_pooling(model_output, attention_mask):
    """
    This function takes the model output and attention masks as
    inputs and performs mean pooling on the token embeddings.
    It calculates the mean of the token embeddings weighted by
    the attention mask and returns the resulting embeddings.
    :param model_output:
    :param attention_mask:
    :return:
    """
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_tokens(encoded_input):
    """
    This function takes encoded input (tokenized and preprocessed) as input,
    passes it through the pre-trained model, and returns the model output,
    which includes token embeddings.
    :param encoded_input:
    :return:
    """
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output


def get_embeddings(lines_list):
    """
    This function takes a list of strings as input, tokenizes and encodes the strings using a pre-trained tokenizer,
    computes their embeddings using the compute_tokens function, normalizes the embeddings,
    and returns the normalized embeddings for the input strings.
    :param lines_list:
    :return:
    """
    encoded_lines = tokenizer(lines_list, padding=True, truncation=True, return_tensors='pt')
    lines_tokens = compute_tokens(encoded_lines)
    lines_embeddings = mean_pooling(lines_tokens, encoded_lines['attention_mask'])
    lines_embeddings = F.normalize(lines_embeddings, p=2, dim=1)
    return lines_embeddings


def get_similar_phrases(trigrams, phrases):
    """
    This function takes a list of trigrams and a list of phrases as inputs.
    For each trigram, it computes the cosine similarity between the trigram embeddings and the phrase embeddings.
    It filters the results based on a similarity score threshold of 0.4, sorts them in descending order,
    and returns a list of tuples containing the trigram index, the trigram, the phrase index,
    and the similarity score for similar trigrams and phrases.
    :param trigrams:
    :param phrases:
    :return:
    """
    phrase_embeddings = get_embeddings(phrases)
    df = pl.DataFrame()
    for tri_index, trigram in enumerate(trigrams):
        trigram_embeddings = get_embeddings(trigram)
        scores = torch.nn.functional.cosine_similarity(trigram_embeddings, phrase_embeddings).numpy()
        trigram_data = {
            'trigram_index': [tri_index]*len(phrases),
            'trigram': [trigram]*len(phrases),
            'phrase_index': [i for i in range(len(phrases))],
            'score': scores
        }
        df = pl.concat([df, pl.DataFrame(trigram_data)])
    df = df.filter(df["score"] >= 0.4).sort("score", descending=True)
    phrase_set = set()
    trigram_set = set()
    new_df = []
    for row in df.rows():
        if row[1] not in trigram_set and row[2] not in phrase_set:
            phrase_set.add(row[2])
            trigram_set.add(row[1])
            new_df.append(row)
    return new_df
