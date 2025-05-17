import torch
import re
import yaml
import selfies as sf

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def dataloader_gen(dataset_dir, percentage, which_vocab, vocab_path,
                   batch_size, PADDING_IDX, shuffle, drop_last=True):
    if which_vocab == "selfies":
        vocab = SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        vocab = RegExVocab(vocab_path)
    elif which_vocab == "char":
        vocab = CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vacab name for configuration which_vocab!")

    dataset = SMILESDataset(dataset_dir, percentage, vocab)

    def pad_collate(batch):
        lengths = [len(x) for x in batch]

        batch = [torch.tensor(x, dtype=torch.long) for x in batch]

        x_padded = pad_sequence(
            batch,
            batch_first=True,
            padding_value=PADDING_IDX
        )

        return x_padded, lengths

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        collate_fn=pad_collate
    )

    return dataloader, len(dataset)

class SMILESDataset(Dataset):
    def __init__(self, smiles_file, percentage, vocab):
        super(SMILESDataset, self).__init__()
        assert(0 < percentage <= 1)

        self.percentage = percentage
        self.vocab = vocab

        if self.vocab.name == "selfies":
            sf.set_semantic_constraints("hypervalent")

        self.data = self.read_smiles_file(smiles_file)
        print("total number of SMILES loaded: ", len(self.data))

        if self.vocab.name == "selfies":
            valid_selfies = []
            for smiles in self.data:
                try:
                    encoded_selfies = sf.encoder(smiles)
                    if encoded_selfies is not None:
                        valid_selfies.append(encoded_selfies)
                except Exception as e:
                    print(f"Error encoding SMILES: {smiles}")
                    print(f"SELFIES EncoderError: {e}")
            self.data = valid_selfies
            print("Total number of valid SELFIES: ", len(self.data))

    def read_smiles_file(self, path):

        with open(path, "r") as f:
            smiles = [line.strip("\n") for line in f.readlines()]

        num_data = len(smiles)

        return smiles[0:int(num_data * self.percentage)]

    def __getitem__(self, index):
        mol = self.data[index]

        mol = self.vocab.tokenize_smiles(mol)

        return mol

    def __len__(self):
        return len(self.data)

class CharVocab:
    def __init__(self, vocab_path):
        self.name = "char"

        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {}
        for token, num in self.vocab.items():
            self.int2tocken[num] = token

        self.tokens = self.vocab.keys()

    def tokenize_smiles(self, smiles):
        n = len(smiles)
        tokenized = ['<sos>']
        i = 0

        while (i < n - 1):

            c2 = smiles[i:i + 2]
            if c2 in self.tokens:
                tokenized.append(c2)
                i += 2
                continue

            c1 = smiles[i]
            if c1 in self.tokens:
                tokenized.append(c1)
                i += 1
                continue

            raise ValueError(
                "Unrecognized charater in SMILES: {}, {}".format(c1, c2))

        if i == n:
            pass
        elif i == n - 1 and smiles[i] in self.tokens:
            tokenized.append(smiles[i])
        else:
            raise ValueError(
                "Unrecognized charater in SMILES: {}".format(smiles[i]))

        tokenized.append('<eos>')

        tokenized = [self.vocab[token] for token in tokenized]
        return tokenized

    def combine_list(self, smiles):
        return "".join(smiles)

class RegExVocab:
    def __init__(self, vocab_path):
        self.name = "regex"

        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {}
        for token, num in self.vocab.items():
            if token == "R":
                self.int2tocken[num] = "Br"
            elif token == "L":
                self.int2tocken[num] = "Cl"
            else:
                self.int2tocken[num] = token

    def tokenize_smiles(self, smiles):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = self.replace_halogen(smiles)
        char_list = re.split(regex, smiles)

        tokenized = ['<sos>']

        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('<eos>')

        tokenized = [self.vocab[token] for token in tokenized]

        return tokenized

    def replace_halogen(self, string):
        br = re.compile('Br')
        cl = re.compile('Cl')
        string = br.sub('R', string)
        string = cl.sub('L', string)

        return string

    def combine_list(self, smiles):
        return "".join(smiles)

class SELFIEVocab:
    def __init__(self, vocab_path):
        self.name = "selfies"

        with open(vocab_path, 'r') as f:
            self.vocab = yaml.full_load(f)

        self.int2tocken = {value: key for key, value in self.vocab.items()}

    def tokenize_smiles(self, mol):
        ints = [self.vocab['<sos>']]

        selfies_list = list(sf.split_selfies(mol))
        for token in selfies_list:
            ints.append(self.vocab[token])

        ints.append(self.vocab['<eos>'])

        return ints

    def combine_list(self, selfies):
        return "".join(selfies)