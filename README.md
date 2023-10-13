# Text Improvement

## Project Description

The **Text Improvement** project focuses on creating a Text Improvement Engine. The objective is to develop a tool that analyzes input text and suggests improvements by comparing it with a set of predefined standard phrases representing ideal articulation of concepts. The project involves pre-loading the tool with standard phrases, utilizing a pre-trained language model to find semantically similar phrases in the input text, and suggesting replacements along with similarity scores.
## Tech overview
This project uses [mpnet](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and cosine similarity word embedding technologies.
The first one was necessary to translate information from text to vectors. And the further one to emphasize the similarity between them.
The ngram analog was used to identify phrases that could be replaced. By using trigrams it is possible to cover almost all suitable phrases that can be replaced by standardized ones. MPnet, namely all-mpnet-base-v2 was used because it is designed for semantic search, has the [best score](https://www.sbert.net/docs/pretrained_models.html#model-overview) for this task according to huggingface and the largest number of downloads. 
For cosine similarity the built-in tool from PyTorch library was used.

Text Processing Pipeline:

suggestion -> trigrams -> embeddings -> max pooling -> normalization -> normalization -> cosine similarity -> sorting -> suggestions  

## Installation

### Pip Dependencies

Ensure you have the following dependencies installed:

```bash
pip install torch==2.1.0 transformers==4.34.0 nltk==3.8.1 numpy==1.26.0 polars==0.19.8
```

## Usage

### Standard Data Usage

To run the project with the standard data provided in the repository, use the following command:

```bash
python main.py
```

This command will work with the predefined raw text and standardized phrases within the repository.

### Custom Data Usage

You can specify paths to raw text, standardized phrases, and output text by adding the following arguments to the CLI:

- `raw_text`: Path to the raw input text file.
- `raw_phrases`: Path to the file containing standardized phrases.
- `output_path`: Path to the output file where improved text will be saved.

Example usage:

```bash
python main.py --raw_text path/to/raw_text.txt --raw_phrases path/to/standardized_phrases.txt --output_path path/to/output.txt
```

## Configuration

- **Python Version**: 3.11

Feel free to modify and adapt the project as needed to suit your requirements. Happy text improving!