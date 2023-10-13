from src.model import get_similar_phrases
import nltk


def generate_trigrams(text):
    """
    This function takes a text input, splits it into words, and generates trigrams from the words.
    Trigrams are sets of three consecutive words. It returns a list of trigrams.
    :param text:
    :return:
    """
    words = text.split()  # Split the input text into words
    trigrams = []

    # Generate trigrams
    for i in range(len(words) - 2):
        trigram = " ".join(words[i:i + 3])  # Extract three consecutive words
        trigrams.append(trigram)  # Add trigram to the list

    return trigrams


def input_new_phrase(sentence, raw_phrases):
    """
    This function takes a sentence and a list of raw phrases as inputs. It generates trigrams for the input sentence,
    finds similar phrases using the get_similar_phrases function from the src.model module,
    and replaces certain trigrams in the sentence with placeholders enclosed in square brackets.
    The placeholders contain the similar phrases along with their similarity scores.
    :param sentence:
    :param raw_phrases:
    :return:
    """
    sentence_trigrams = generate_trigrams(sentence)
    similar_phrases = get_similar_phrases(sentence_trigrams, raw_phrases)
    words = sentence.split()
    counter = 0

    for phrase in similar_phrases:
        text_phrase = raw_phrases[phrase[2]]
        score = round(phrase[3], 3)
        index = phrase[0] + 3 + counter
        counter += 1
        new_string = "{" + text_phrase + "}" + " score: " + str(score)
        words.insert(index, new_string)
        words.insert(index, "]")
        words.insert(index-3, "[")
        index += 2

    new_sentence = " ".join(words)

    return new_sentence


def improve_text(raw_text, raw_phrases):
    """
    This function takes raw text and a list of raw phrases as inputs.
    It tokenizes the raw text into sentences, calls the input_new_phrase function for each sentence to improve it,
    and concatenates the improved sentences into a new text. The function returns the improved text where certain
    phrases are replaced with placeholders along with their similarity scores.
    :param raw_text:
    :param raw_phrases:
    :return:
    """
    text_list = nltk.sent_tokenize(raw_text)
    new_sentences = []

    for sentence in text_list:
        new_sentences.append(input_new_phrase(sentence, raw_phrases))

    new_text = " ".join(new_sentences)

    return new_text
