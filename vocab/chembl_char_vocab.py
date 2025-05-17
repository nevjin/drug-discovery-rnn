"""generate the vocabulary accorrding to the regular expressions of
SMILES of molecules."""
import yaml
from tqdm import tqdm


def read_smiles_file(path, percentage):
    with open(path, 'r') as f:
        smiles = [line.strip("\n") for line in f.readlines()]
    num_data = len(smiles)
    return smiles[0:int(num_data * percentage)]


def tokenize(smiles, tokens):
    """
    Takes a SMILES string and returns a list of tokens.
    Atoms with 2 characters are treated as one token. The 
    logic references this code piece:
    https://github.com/topazape/LSTM_Chem/blob/master/lstm_chem/utils/smiles_tokenizer2.py
    """
    n = len(smiles)
    tokenized = []
    i = 0

    # process all characters except the last one
    while (i < n - 1):
        # procoss tokens with length 2 first
        c2 = smiles[i:i + 2]
        if c2 in tokens:
            tokenized.append(c2)
            i += 2
            continue

        # tokens with length 2
        c1 = smiles[i]
        if c1 in tokens:
            tokenized.append(c1)
            i += 1
            continue

        raise ValueError(
            "Unrecognized charater in SMILES: {}, {}".format(c1, c2))

    # process last character if there is any
    if i == n:
        pass
    elif i == n - 1 and smiles[i] in tokens:
        tokenized.append(smiles[i])
    else:
        raise ValueError(
            "Unrecognized charater in SMILES: {}".format(smiles[i]))
    return tokenized


if __name__ == "__main__":
    dataset_dir = "../../chembl-data/chembl_28/chembl_28_sqlite/chembl28-cleaned.smi"
    output_vocab = "./chembl_char_vocab.yaml"

    atoms = [
        'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
        'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te'
    ]

    special = [
        '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
        '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 'p', 's'
    ]

    tokens = atoms + special
    tokens = set(tokens)

    print("computing token set from dataset...")
    smiles = read_smiles_file(dataset_dir, 1)
    data_tokens = []
    [data_tokens.extend(tokenize(x, tokens)) for x in tqdm(smiles)]
    data_tokens = set(data_tokens)

    print("validating token set from dataset...")
    assert(data_tokens.issubset(tokens))
    print("OK")

    vocab_dict = {}
    for i, token in enumerate(tokens):
        vocab_dict[token] = i

    i += 1
    vocab_dict['<eos>'] = i
    i += 1
    vocab_dict['<sos>'] = i
    i += 1
    vocab_dict['<pad>'] = i

    with open(output_vocab, 'w') as f:
        yaml.dump(vocab_dict, f)
