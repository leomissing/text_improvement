from src.utils import *
from src.text_processing import improve_text
import argparse

# Parser specs and ops
parser = argparse.ArgumentParser(description="Here is my script")
parser.add_argument('raw_text', nargs='?', help="Path to raw text file", default="data/raw/sample_text.txt")
parser.add_argument('raw_phrases', nargs='?', help="Path to standardised pharses",
                    default="data/raw/Standardised terms.csv")
parser.add_argument('output_path', nargs='?', help="Save path", default="data/processed/output.txt")
args = parser.parse_args()

print(
    f'\nInput arguments: \nRaw text path - {args.raw_text} \n'
    f'Raw phrases path - {args.raw_phrases} \nOutput path - {args.output_path}')
print("-" * 100)

raw_text = read_txt(args.raw_text)
raw_phrases = read_csv(args.raw_phrases)

print('Files has been read.')
print("-" * 100)

processed_text = improve_text(raw_text=raw_text, raw_phrases=raw_phrases)

print('Text has been improved')
print("-" * 100)

with open(args.output_path, 'w') as file:
    file.write(processed_text)

print(f"Text has been saved to '{args.output_path}'")
print("-" * 100)


def main():
    pass


if __name__ == "__main__":
    main()
