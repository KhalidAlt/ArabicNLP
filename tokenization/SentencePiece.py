
import torch
import torch.nn as nn
import numpy as np
import copy



import tokenizers
from tokenizers import ByteLevelBPETokenizer,BertWordPieceTokenizer ,SentencePieceBPETokenizer,decoders,SentencePieceUnigramTokenizer #,CharBPETokenizer
from tokenizers.pre_tokenizers import Metaspace 
from tokenizers.decoders import ByteLevel
from tokenizers.normalizers import NFKC,BertNormalizer
from tokenizers import normalizers


from pathlib import Path


def download_Dataset():
    """Download the dataset and extract the path

    Returns
    -------
    paths
        a string that holds the path of the dataset
    """

    paths = [str(x) for x in Path("../dataset/").glob("*.txt")] 
    
    return paths




def train_tokenizer(tokenizer,paths):
    """train the tokenizer using huggingface tokenizers algorithm

    Parameters
    ----------
    tokenizer : tokenizers.implementations
        A tokenizer implementation that is not trained yet 
    paths : list
        a list that holds the path of the dataset
    strr : str, optional
        a string name that used as base name for the saved dictionary 
    Returns
    -------
    tokenizer
        a trained tokenizer 
    """
    #min_frequency=2
    tokenizer.train(files=paths,min_frequency=2, vocab_size=32_000,special_tokens=[
      "<eos>", 
      "<pad>",
      "<bos>",
      "<unk>",
      "<mask>",
    ])


  
    return tokenizer



paths=download_Dataset()


# build Bertnormlizer to remove /n from the texts 

bert_normalizer = BertNormalizer(clean_text=True, handle_chinese_chars=False)

# Add bert_normlizer to a normlizer pipline including NFKC based normlizer that used by SentencePiece

normalizer = normalizers.Sequence([bert_normalizer, normalizers.NFKC()])

# Construct SentencePiece tokenizer that use BPE learning algorithm

tokenizer = SentencePieceBPETokenizer()

# add the normalizer pipline created before to the sentencePiec tokenizer pipline

tokenizer._tokenizer.normalizer=normalizer

# Train the tokenizer
trained_tokenizer=train_tokenizer(tokenizer,paths)

# Save the trained tokenizer in the following folder

trained_tokenizer.save_model("arabic")
