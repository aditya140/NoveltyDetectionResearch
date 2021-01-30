import spacy
from collections import Counter
import sys
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import unicodedata
import re
from transformers import DistilBertTokenizer
import pickle
from utils.statics import CHAR_LIST


char_to_ind = {CHAR_LIST[j]: j for j in range(len(CHAR_LIST))}
ind_to_char = {j: CHAR_LIST[j] for j in range(len(CHAR_LIST))}


class LanguageIndex:
    def __init__(self, text=None, config=None):

        """
        Lanugae Index -  Langugae Index  acts as an interface (similar to torch.vocab) for converting
        text to corresponding tokens, and also has the capability of indexing the vocabulaty by providing
        the phrase exerpts form the languge to index.

        Args:
            text ([list], optional): List of sentences to be indexed. Defaults to None.
            config ([type], optional): . Defaults to None.
        """
        self.text = text
        self.config = config

        if self.config == None:
            self.config = LangConf()

        self.char_emb = self.config.char_emb
        self.char_emb_max_len = self.config.max_char_len

        self.lower = self.config.lower_case
        self.tokenizer_ = self.config.tokenizer
        self.max_len = self.config.max_len

        if self.tokenizer_ == "BERT" or self.tokenizer_ == "Distil_BERT":
            self.load_bert()
            self.tokenize = self.tokenize_bert
            self.encode = self.encoder_bert
            self.decode = self.decode_bert
        elif self.tokenizer_ == "spacy":
            self.load_spacy()
            self.tokenize = self.tokenize_spacy
            self.encode = self.encode_base
            self.decode = self.decode_base
            self.create_language()
        else:
            pass

    def load_bert(self):
        """
        Load BERT tokenizer
        """
        if self.tokenizer_ == "BERT":
            model_type = "bert-base-uncased"
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_type)
        if self.tokenizer_ == "Distil_BERT":
            model_type = "distilbert-base-uncased"
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(model_type)
        self.vocab_size = self.bert_tokenizer.vocab_size

    def load_spacy(self):
        """
        Load Spacy
        """
        self.spacy = spacy.load("en")

    def load_tokenizer(self, tok):
        """
        Load external tokenizer

        Args:
            tok ([type]): Tokenizer to load
        """
        self.bert_tokenizer = tok
        self.vocab_size = self.bert_tokenizer.vocab_size

    def create_language(self):
        """
        Create the vocab from the given text exceprt
        """
        self.lower = self.config.lower_case
        self.word2idx = {}
        self.idx2word = {}
        self.special = {}
        self.vocab_size = (
            self.config.vocab_size - 4
            if self.config.vocab_size != None
            else sys.maxsize
        )

        # add a padding token with index 0, init token as index 1, eos token as 2, unk token as 3
        self.word2idx[self.config.pad] = 0
        self.special["pad_token"] = self.config.pad
        self.word2idx[self.config.init_token] = 1
        self.special["init_token"] = self.config.init_token
        self.word2idx[self.config.eos_token] = 2
        self.special["eos_token"] = self.config.eos_token
        self.word2idx[self.config.unk_token] = 3
        self.special["unk_token"] = self.config.unk_token

        self.vocab = set()
        self.counter = Counter()
        self.create_index()

    def create_index(self):
        for phrase in self.text:
            # update with individual tokens
            tokens = self.tokenize(phrase.lower() if self.lower else phrase)
            self.vocab.update(tokens)
            self.counter.update(tokens)

        # sort the vocab
        self.vocab = sorted(self.vocab)
        start_index = max(self.word2idx.values()) + 1

        # word to index mapping
        for index, word in enumerate(self.counter.most_common(self.vocab_size)):
            self.word2idx[word[0]] = index + start_index

        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
        del self.text

    def tokenize_spacy(self, phrase):
        """Tokenize using spacy tokenizer

        Args:
            phrase ([type]): Phrase to tokenzie

        Returns:
            [list] : Tokens
        """
        return [tok.text for tok in self.spacy.tokenizer(phrase)]

    def tokenize_bert(self, phrase):
        """Tokenize using BERT tokenizer

        Args:
            phrase ([type]): Phrase to tokenzie

        Returns:
            [list] : Tokens
        """
        return self.bert_tokenizer.tokenize(phrase)

    def tokenize_base(self, phrase):
        """Basic tokenizer

        Args:
            phrase ([type]): Phrase to tokenzie

        Returns:
            [list] : Tokens
        """
        return self.preprocess(phrase)

    def encoder_bert(self, input_, special_tokens=True):
        """Encode input string using bert tokenizer

        Args:
            input_ ([string]): input string to be encoded
            special_tokens (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: encoded list
        """
        return self.bert_tokenizer.encode(
            input_, padding="max_length", max_length=self.max_len
        )[: self.max_len]

    def encode_base(self, input_, special_tokens=True):
        """Encode input string using spacy tokenizer

        Args:
            input_ ([string]): input string to be encoded
            special_tokens (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: encoded list
        """
        return self.lang_encode(input_, special_tokens=special_tokens)

    def decode_bert(self, input_, to_string=False):
        """Decode input string using bert encoder

        Args:
            input_ ([type]): Input tokens to be decoded
            to_string (bool, optional): Converts to one single string if True. Defaults to False.

        Returns:
            [type]: [description]
        """
        if to_string:
            return self.bert_tokenizer.convert_tokens_to_string(input_)
        else:
            return self.bert_tokenizer.convert_ids_to_tokens(input_)

    def decode_base(self, input_, to_string=False):
        """Decode input string using spacy encoder

        Args:
            input_ ([type]): Input tokens to be decoded
            to_string (bool, optional): Converts to one single string if True. Defaults to False.

        Returns:
            [type]: [description]
        """
        return self.lang_decode(input_, to_string)

    def encode_batch(self, batch, special_tokens=True, pair=False):
        """
        Encode batch of text

        Args:
            batch ([type]): [description]
            special_tokens (bool, optional): [description]. Defaults to True.
            pair (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if pair:
            return np.array(
                np.array(
                    [
                        self.encoder_pair(s1, s2, special_tokens=special_tokens)
                        for s1, s2 in zip(*batch)
                    ]
                )
            )
        else:
            return np.array(
                np.array(
                    [self.encode(obj, special_tokens=special_tokens) for obj in batch]
                )
            )

    def decode_batch(self, batch):
        """Decode batch of tokens

        Args:
            batch ([type]): [description]

        Returns:
            [type]: [description]
        """
        return [self.decode(obj) for obj in batch]

    def lang_encode(self, input_, special_tokens=True):
        """Basic implementation of sentence encoder

        Args:
            input_ ([type]): Sentence
            special_tokens (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: Tokens
        """
        pad_len = self.max_len
        input_ = input_.lower() if self.lower else input_
        tokens = [tok for tok in self.tokenize(input_)]
        if pad_len != None:
            pad_arr = [0] * (pad_len - len(tokens) - (2 if special_tokens else 0))
        return (
            ([1] if special_tokens else [])
            + (
                [self.word2idx[s] if s in self.word2idx.keys() else 3 for s in tokens]
                + pad_arr
            )[: (pad_len - 2 if special_tokens else pad_len)]
            + ([2] if special_tokens else [])
        )

    def encoder_pair(self, sent1, sent2, special_tokens=True):
        """Encode pair of sentences separated by SEP token

        Args:
            sent1 ([string]): Sentence 1
            sent2 ([string]): Sentence 2
            special_tokens (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        inp_ = sent1 + " " + self.bert_tokenizer.sep_token + " " + sent2
        return self.encode(inp_, special_tokens=special_tokens)

    def lang_decode(self, input_, to_string=False):
        """
        Decode a list of tokens

        Args:
            input_ ([list]): list of tokens
            to_string (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        sent = [
            self.idx2word[s] if s in self.idx2word.keys() else self.special["unk_token"]
            for s in input_
        ]
        if to_string:
            return " ".join(sent)
        return sent

    def char_embedd(self, token_list):
        """[summary]

        returns a char embedding of token list

        Args:
            token_list ([type]): [description]
        """
        sent_vec = []
        for tok in token_list:
            if len(tok)<self.char_emb_max_len:
                pad = [0]**(self.char_emb_max_len-len(tok))
            else:
                pad = []
            tok_vec = [char_to_ind[i] for i in tok]+pad
            sent_vec.append(tok_vec[:self.char_emb_max_len])
        return sent_vec
        

    def vocab_size_final(self):
        if self.tokenizer_ == "BERT" or self.tokenizer_ == "Distil_BERT":
            return self.vocab_size
        return len(self.word2idx.keys())

    def unicode_to_ascii(self, s):
        """
        Normalizes latin chars with accent to their canonical decomposition
        """
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.rstrip().strip()
        return w


def read_glove():
    with open("glove.pkl", "rb") as f:
        glove_model = pickle.load(f)
    return glove_model


def read_embedding_file(vocab):
    glove_model = read_glove()
    matrix_len = len(vocab.word2idx)
    weights_matrix = np.zeros((matrix_len, 300))
    words_found = 0
    print("creating embedding matrix")
    for i, word in enumerate(vocab.idx2word.items()):
        try:
            weights_matrix[i] = glove_model.wv[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))
    return weights_matrix


class LangConf:
    """ Base Language Index Config """

    tokenizer = "BERT"
    max_len = 100
    vocab_size = None
    lower_case = True
    char_emb = True
    max_char_len = 9

    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class BertLangConf(LangConf):
    """ Bert Index Config """

    def __init__(self, vocab_size, **kwargs):
        super(BertLangConf, self).__init__(vocab_size, **kwargs)


class GloveLangConf(LangConf):
    """ Glove Index Config """

    tokenizer = "spacy"
    pad = "<PAD>"
    init_token = "<SOS>"
    eos_token = "<EOS>"
    unk_token = "<UNK>"

    def __init__(self, vocab_size, **kwargs):
        super(GloveLangConf, self).__init__(vocab_size, **kwargs)
