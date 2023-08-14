import json
import string
from typing import Iterable


class BOT:
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.special_token = 'Ġ'
        self.unknown = 0
        self.vocab = [self.special_token] + [i for i in string.ascii_lowercase + string.punctuation + string.digits]
        self.merges = []  # dictionary of word fusions ("Ġ", "t"): "Ġt"

    def _create_vocab(self, word_frequencies):
        print("Split words...")
        splits = self._get_splits(word_frequencies)

        while len(self.vocab) < self.vocab_size:
            print("Compute pairs word_frequencies...")
            pair_freq = self._compute_pairs_freq(splits, word_frequencies)  # frequency of each pair in the corpus
            best_pair, max_freq = self._find_best_pair(pair_freq)  # find the most frequent pair

            print("Refactor splits by the most frequencies pair...")
            # merge of the most frequent character pair
            splits = self._merge_pair(*best_pair, splits, word_frequencies)

            print("Update vocab...")
            self.merges.append((best_pair, "".join(best_pair)))  # a merger to be learned
            self.vocab.append("".join(best_pair))  # add to the vocabulary
            print(f"Vocab: {len(self.vocab)} / {self.vocab_size}")

    def _compute_frequencies_words(self, text_corpus: Iterable[str]) -> list[list[str, int]]:
        word_frequencies = []

        for text in text_corpus:
            preprocess_text = [i for i in text.lower().split()]
            preprocess_text = [preprocess_text[0]] + [self.vocab[0] + i for i in preprocess_text[1:]]

            words = [i[0] for i in word_frequencies]
            for word in preprocess_text:
                if word in words:
                    word_frequencies[words.index(word)][1] += 1
                else:
                    word_frequencies.append([word, 1])
            del words
        return word_frequencies

    @staticmethod
    def _get_splits(word_frequencies) -> dict:
        splits = {}
        for word in [word[0] for word in word_frequencies]:
            splits[word] = [letter for letter in word]
        return splits

    @staticmethod
    def _compute_pairs_freq(splits, word_frequencies) -> list[list[tuple[str, str], int]]:
        pair_freq = []

        for word, freq in word_frequencies:
            split = splits[word]

            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])

                list_pairs = [p[0] for p in pair_freq]
                if pair in list_pairs:
                    pair_freq[list_pairs.index(pair)][1] += freq
                else:
                    pair_freq.append([pair, freq])

        return pair_freq

    @staticmethod
    def _find_best_pair(pairs_freq):
        best_pair = ""
        max_freq = None

        for pair, freq in pairs_freq:
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq

        return best_pair, max_freq

    @staticmethod
    def _merge_pair(a, b, splits, word_frequencies):
        for word in [word[0] for word in word_frequencies]:
            split = splits[word]

            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2:]
                else:
                    i += 1

            splits[word] = split

        return splits

    def fit(self, text_corpus: Iterable[str]):
        print("Compute word frequencies...")
        word_frequencies = self._compute_frequencies_words(text_corpus)  # word frequency in the corpus
        self._create_vocab(word_frequencies)

    def decode(self, text: str) -> list[int]:
        word_frequencies = self._compute_frequencies_words([text])
        splits = [[letter for letter in word] for word in [word[0] for word in word_frequencies]]

        for pair, merge in self.merges:
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        tokenized_text = sum(splits, [])
        tokens = []
        for token in tokenized_text:
            try:
                token_index = self.vocab.index(token)
                tokens.append(token_index)
            except:
                tokens.append(self.unknown)
        return tokens

    def encode(self, tokens: list[int]):
        text = [self.vocab[i] for i in tokens]
        return "".join(text).replace(self.special_token, ' ')

    def load(self):
        with open("merges.json", encoding="utf-8") as f:
            load_merges = json.load(f)
            self.merges = load_merges

        with open("vocab.json", encoding="utf-8") as f:
            load_vocab = json.load(f)
            self.vocab = load_vocab

    def save(self):
        with open("merges.json", 'w') as f:
            f.write(json.dumps(self.merges))

        with open("vocab.json", 'w') as f:
            f.write(json.dumps(self.vocab))