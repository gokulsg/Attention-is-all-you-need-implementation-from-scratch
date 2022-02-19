from collections import Counter
from torchtext.vocab import Vocab
import io
import spacy

train_filepaths_de = './data/PHP.de-en.de'
train_filepaths_en = './data/PHP.de-en.en'

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenize German sentence to list of words
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenize English sentence to list of words
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def build_vocab(filepath, tokenizer):
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for sentence in f:
            counter.update(tokenizer(sentence))
    counter.update('<unk>')
    counter.update('<pad>') 
    counter.update('<sos>') 
    counter.update('<eos>') 
    return Vocab(counter)

# German and English Vocabularies #
trg_field = build_vocab(train_filepaths_de, tokenize_de)
src_field = build_vocab(train_filepaths_en, tokenize_en)