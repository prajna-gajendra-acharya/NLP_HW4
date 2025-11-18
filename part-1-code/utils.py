import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

_QWERTY_NEIGHBORS = {
    'q':'was', 'w':'qesad', 'e':'wsdr', 'r':'edft', 't':'rfgy', 'y':'tghu', 'u':'yjhki', 'i':'ujklo',
    'o':'iklp', 'p':'ol',
    'a':'qwsz', 's':'weadzx', 'd':'ersfcx', 'f':'rtdgcv', 'g':'tfhbv', 'h':'ygjnb', 'j':'uikhmn',
    'k':'ijolm', 'l':'okp',
    'z':'asx', 'x':'zsdc', 'c':'xdfv', 'v':'cfgb', 'b':'vghn', 'n':'bhjm', 'm':'njk'
}

def _neighbor(c):
    pool = _QWERTY_NEIGHBORS.get(c.lower(), "")
    return random.choice(pool) if pool else c

def _noisify_word(word, p_sub=0.12, p_del=0.02, p_ins=0.02):
    # Keep first/last char stable to avoid flipping the word entirely
    if len(word) <= 2 or not word.isalpha():
        return word
    chars = list(word)
    i = 1
    out = [chars[0]]
    while i < len(chars) - 1:
        ch = chars[i]
        r = random.random()
        if r < p_del:
            # delete this char (skip append)
            pass
        elif r < p_del + p_ins:
            # insert a neighbor before current char
            out.append(_neighbor(ch))
            out.append(ch)
        elif r < p_del + p_ins + p_sub:
            # substitute with neighbor
            out.append(_neighbor(ch))
        else:
            out.append(ch)
        i += 1
    out.append(chars[-1])
    # Preserve original casing style crudely (title/allcaps/else)
    noisy = "".join(out)
    if word.isupper():
        return noisy.upper()
    if word.istitle():
        return noisy.capitalize()
    return noisy.lower()  # encourage casual casing

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]

    # light punctuation normalization: sometimes drop commas/periods; remove some apostrophes
    # (keeps sentences readable but slightly “messier”)
    text = re.sub(r"[’']", lambda m: "" if random.random() < 0.35 else m.group(0), text)
    text = re.sub(r",", lambda m: "" if random.random() < 0.25 else m.group(0), text)
    text = re.sub(r"\.\s", lambda m: " " if random.random() < 0.15 else m.group(0), text)

    tokens = word_tokenize(text)
    new_tokens = []
    for tok in tokens:
        if tok.isalpha():
            new_tokens.append(_noisify_word(tok))
        else:
            # occasionally drop doubled punctuation like "!!"
            if re.fullmatch(r"[!?]{2,}", tok) and random.random() < 0.3:
                tok = tok[0]
            new_tokens.append(tok)

    transformed = TreebankWordDetokenizer().detokenize(new_tokens)

    # occasional extra space collapse & casual all-lowercase bias
    transformed = re.sub(r"\s+", " ", transformed).strip()
    if random.random() < 0.2:
        transformed = transformed.lower()

    example["text"] = transformed

    ##### YOUR CODE ENDS HERE ######

    return example
