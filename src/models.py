import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, Dropout, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.utils import load_data, load_tokenizer, clean_sentences, \
    get_words_max_length, clean_sentence, encode_sentences, pad_sentences


INTENT_CLSF_STOPWORDS = [
    'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'this', 'that', "that'll", 'these', 'those', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'just', 'now', 'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain'
]


class IntentClassifier(object):

    def __init__(self,
                 weights_file_path,
                 training_set_file_path,
                 input_tokenizer_file_path,
                 output_tokenizer_file_path):

        self._weights_file_path = weights_file_path
        self._training_set_file_path = training_set_file_path
        self._input_tokenizer_file_path = input_tokenizer_file_path
        self._output_tokenizer_file_path = output_tokenizer_file_path

        self._lemmatizer = nltk.wordnet.WordNetLemmatizer()
        self._clean_policy = '^ a-z A-Z 0-9 @#'

        # set attributes depending on the training set used
        self._input_word_tokenizer = None
        self._output_label_tokenizer = None
        self._vocab_size = None
        self._max_length = None
        self._nr_intent_classes = None
        self._load_tokenizers()

        self.model = self.create_model()

    def _load_tokenizers(self):

        df = load_data(self._training_set_file_path)

        cleaned_questions = clean_sentences(df['question'],
                                            stopwords=INTENT_CLSF_STOPWORDS,
                                            clean_policy=self._clean_policy,
                                            lemmatizer=self._lemmatizer)

        self._input_word_tokenizer = load_tokenizer(
            self._input_tokenizer_file_path
        )

        self._vocab_size = len(self._input_word_tokenizer.word_index) + 1
        self._max_length = get_words_max_length(cleaned_questions)

        self._output_label_tokenizer = load_tokenizer(
            self._output_tokenizer_file_path
        )
        self._nr_intent_classes = len(self._output_label_tokenizer.word_index)

    def create_model(self):
        model = Sequential()
        model.add(Embedding(self._vocab_size,
                            128,
                            input_length=self._max_length,
                            trainable=False))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self._nr_intent_classes, activation="softmax"))

        return model

    def load_weights(self):
        self.model.load_weights(self._weights_file_path)

    def predict(self, text):

        cleaned_text = clean_sentence(text,
                                      stopwords=INTENT_CLSF_STOPWORDS,
                                      clean_policy=self._clean_policy,
                                      lemmatizer=self._lemmatizer)

        encoded_text = encode_sentences(self._input_word_tokenizer,
                                        cleaned_text)

        # Check for unknown words
        if [] in encoded_text:
            encoded_text = list(filter(None, encoded_text))
        out = np.array(encoded_text).reshape(1, len(encoded_text))

        x = pad_sentences(out, self._max_length)

        pred = self.model.predict_proba(x)[0]

        intent_index = np.argmax(pred) + 1
        confidence = round(np.max(pred), 3)
        predicted_intent = self._output_label_tokenizer.index_word[intent_index]

        return predicted_intent, confidence

    # def clean_sentences(self, sentences):
    #     words = []
    #     for s in sentences:
    #         clean_sentence = self._clean_sentence(s)
    #         words.append(clean_sentence)
    #     return words

    # def _pad_doc(self, encoded_doc):
    #     return pad_sequences(encoded_doc,
    #                          maxlen=self._max_length,
    #                          padding='post')

    # def _clean_sentence(self, sentence):
    #     # removed every punctuation and special characters.
    #     clean = re.sub(r'[{}]'.format(self._clean_policy), " ", sentence)
    #
    #     # tokenize
    #     tokens = word_tokenize(clean)  # needs downloaded punkt
    #
    #     # remove stopwords
    #     tokens = filter(lambda x: x not in INTENT_CLSF_STOPWORDS, tokens)
    #
    #     # lemmatize
    #     return [self._lemmatizer.lemmatize(i.lower()) for i in tokens]

    # def _create_tokenizer(self, words, filters):
    #     """Creates a tokenizer objects that encodes all the questions by
    #     mapping each token to an integer"""
    #     token = Tokenizer(filters=filters)
    #     token.fit_on_texts(words)
    #     return token
    #
    # def _get_words_max_length(self, words):
    #     """Get the length of the maximum length word"""
    #     return len(max(words, key=len))