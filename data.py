import os
from io import open
import torch
import random

def split_qa_data(data_dir, qa_filename='qa_dataset.txt'):
    qa_file = os.path.join(data_dir, qa_filename)
    if not os.path.exists(qa_file):
        return

    with open(qa_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Group lines into QA blocks
    blocks = []
    current_block = []
    for line in lines:
        if line.strip().startswith('Q:') or line.strip().startswith('A:'):
            current_block.append(line)
            if line.strip().startswith('A:'):
                blocks.append("".join(current_block))
                current_block = []

    random.seed(42)
    random.shuffle(blocks)

    total = len(blocks)
    train_split = int(0.8 * total)
    valid_split = int(0.9 * total)

    train_blocks = blocks[:train_split]
    valid_blocks = blocks[train_split:valid_split]
    test_blocks = blocks[valid_split:]

    with open(os.path.join(data_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_blocks)
    with open(os.path.join(data_dir, 'valid.txt'), 'w', encoding='utf-8') as f:
        f.writelines(valid_blocks)
    with open(os.path.join(data_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.writelines(test_blocks)

    print(f"Split {total} QA pairs into:")
    print(f"Train: {len(train_blocks)}")
    print(f"Valid: {len(valid_blocks)}")
    print(f"Test: {len(test_blocks)}")


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, qa_format=False):
        if qa_format and os.path.exists(os.path.join(path, 'qa_dataset.txt')):
            split_qa_data(path)
            
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'), qa_format)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), qa_format)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), qa_format)

    def tokenize(self, path, qa_format=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        if qa_format:
            qa_pairs = []
            current_q = ""
            # Parse file into Q & A pairs
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Q:"):
                        current_q = line
                    elif line.startswith("A:") and current_q:
                        qa_pairs.append(current_q + " " + line + " <eos>")
                        current_q = ""
            
            # Add words to the dictionary
            for pair in qa_pairs:
                words = pair.split()
                for word in words:
                    self.dictionary.add_word(word)
            
            # Tokenize file content
            idss = []
            for pair in qa_pairs:
                words = pair.split()
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            if not idss:
                return torch.tensor([]).type(torch.int64)
            return torch.cat(idss)
        else:
            # Original Tokenize logic
            with open(path, 'r', encoding="utf8") as f:
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        self.dictionary.add_word(word)

            with open(path, 'r', encoding="utf8") as f:
                idss = []
                for line in f:
                    words = line.split() + ['<eos>']
                    ids = []
                    for word in words:
                        ids.append(self.dictionary.word2idx[word])
                    idss.append(torch.tensor(ids).type(torch.int64))
                ids = torch.cat(idss)

            return ids
