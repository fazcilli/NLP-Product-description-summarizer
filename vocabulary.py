from typing import List

import torch
from torch.utils.data import Dataset

class Vocabulary():
    """ Object holding vocabulary and mappings
    Args:
        word_list: ``list`` A list of words. Words assumed to be unique.
        add_unk_token: ``bool` Whether to create an token for unknown tokens.
    """
    def __init__(self, word_list, add_unk_token=False):
        self._pad_token = '<pad>'
        self._unk_token = '<unk>' if add_unk_token else None

        self._special_tokens = [self._pad_token]
        if self._unk_token:
            self._special_tokens += [self._unk_token]

        self.word_list = word_list

    def __len__(self):
        return len(self._token_to_id)

    @property
    def special_tokens(self):
        return self._special_tokens

    @property
    def pad_token_id(self):
        return self.map_token_to_id(self._pad_token)

    @property
    def word_list(self):
        return self._word_list

    @word_list.setter
    def word_list(self, wl):
        self._word_list = wl
        self._init_vocab()

    def _init_vocab(self):
        self._id_to_token = self._word_list + self._special_tokens
        self._token_to_id = {token: id for id, token in
                             enumerate(self._id_to_token)}

    def map_token_to_id(self, token: str):
        """ Maps a single token to its token ID """
        if token not in self._token_to_id:
            token = self._unk_token
        return self._token_to_id[token]

    def map_id_to_token(self, id: int):
        """ Maps a single token ID to its token """
        return self._id_to_token[id]

    def map_tokens_to_ids(self, tokens: List[str], max_length: int = None):
        """ Maps a list of tokens to a list of token IDs """
        # truncate extra tokens and pad to `max_length`
        if max_length:
            tokens = tokens[:max_length]
            tokens = tokens + [self._pad_token]*(max_length-len(tokens))
        return [self.map_token_to_id(token) for token in tokens]

    def map_ids_to_tokens(self, ids: List[int], filter_padding=True):
        """ Maps a list of token IDs to a list of token """
        tokens = [self.map_id_to_token(id) for id in ids]
        if filter_padding:
            tokens = [t for t in tokens if t != self._pad_token]
        return tokens