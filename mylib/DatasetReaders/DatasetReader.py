from typing import Dict
import json
import logging
import glob
import torch
import random
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers import Tokenizer, CharacterTokenizer, WordTokenizer
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]



@DatasetReader.register("names")
class NamesDatasetReader(DatasetReader):

        def __init__(self, lazy: bool = False) -> None:

            super().__init__(lazy)
            self._c_tokenizer =  CharacterTokenizer()

            self._c_token_indexers = {"tokens": SingleIdTokenIndexer()}


        @overrides
        def text_to_instance(self, language: str, name: str) -> Instance:  # type: ignore
            # pylint: disable=arguments-differ

            tokenized_name = self._c_tokenizer.tokenize(name)

            name_field = TextField(tokenized_name, self._c_token_indexers)

            fields = {'name': name_field}

            if language is not None:
                print('language',language)
                fields['label'] = LabelField(language)

            return Instance(fields)

        @overrides
        def _read(self, file_path : str = './data/names/*.txt'):

            all_filenames = glob.glob('./data/names/*.txt')

            category_lines = {}
            all_categories = []

            for filename in all_filenames:
                category = filename.split('/')[-1].split('.')[0]
                all_categories.append(category)
                lines = readLines(filename)
                category_lines[category] = lines

            def random_training_pair():
                category = random.choice(all_categories)
                line = random.choice(category_lines[category])
                return category, line

            length_dict = sum(len(value) for key, value in category_lines.items())

            for i in range(len(category_lines)):
                 language, name = random_training_pair()
                 yield self.text_to_instance(language, name)
