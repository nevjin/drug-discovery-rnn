# Combined Pretraining and Fine-Tuning
import yaml
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from rdkit import Chem
import selfies as sf

from dataloader import dataloader_gen, SELFIEVocab, RegExVocab, CharVocab
from model import RNN

from rdkit.Chem import rdMolDescriptors, DataStructs

# suppress rdkit error
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def make_vocab(config):
    which_vocab = config["which_vocab"]
    vocab_path = config["vocab_path"]
    if which_vocab == "selfies":
        return SELFIEVocab(vocab_path)
    elif which_vocab == "regex":
        return RegExVocab(vocab_path)
    elif which_vocab == "char":
        return CharVocab(vocab_path)
    else:
        raise ValueError("Wrong vocab name for configuration which_vocab!")

def sample(model, vocab, batch_size):
    model.eval()
    sampled_ints = model.sample(batch_size=batch_size, vocab=vocab, device=device)
    molecules = []
    sampled_ints = sampled_ints.tolist()
    for ints in sampled_ints:
        molecule = []
        for x in ints:
            if vocab.int2tocken[x] == '<eos>':
                break
            else:
                molecule.append(vocab.int2tocken[x])
        molecules.append("".join(molecule))
    if vocab.name == 'selfies':
        molecules = [sf.decoder(x) for x in molecules]
    return molecules

def compute_valid_rate(molecules):
    num_valid, num_invalid = 0, 0
    for mol in molecules:
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            num_invalid += 1
        else:
            num_valid += 1
    return num_valid, num_invalid

def fine_tune_model(model, dataloader, optimizer, scheduler, loss_function, vocab, fine_tune_epochs, out_dir):
    print("Starting fine-tuning...")
    for epoch in range(1, fine_tune_epochs + 1):
        model.train()
        train_loss = 0
        for data, lengths in tqdm(dataloader):
            lengths = [length - 1 for length in lengths]
            optimizer.zero_grad()
            data = data.to(device)
            preds = model(data, lengths)
            targets = pack_padded_sequence(
                data[:, 1:], lengths, batch_first=True, enforce_sorted=False
            ).data
            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Fine-tuning Epoch {epoch}, Loss: {train_loss:.4f}")
        scheduler.step(train_loss)
        sampled_molecules = sample(model, vocab, batch_size=1024)
        num_valid, num_invalid = compute_valid_rate(sampled_molecules)
        valid_rate = num_valid / (num_valid + num_invalid)
        print(f"Fine-tuning Epoch {epoch}, Valid Rate: {valid_rate:.4f}")
        if (epoch) % 50 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"fine_tuned_model_epoch{epoch}.pt"))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    config_dir = "./train.yaml"
    with open(config_dir, 'r') as f:
        config = yaml.full_load(f)

    out_dir = config['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    pretrained_model_dir = os.path.join(out_dir, 'fine_tuned_model_epoch10.pt')

    dataset_dir = config['dataset_dir']
    which_vocab = config['which_vocab']
    vocab_path = config['vocab_path']
    percentage = config['percentage']

    batch_size = config['batch_size']
    shuffle = config['shuffle']
    PADDING_IDX = config['rnn_config']['num_embeddings'] - 1
    dataloader, train_size = dataloader_gen(
        dataset_dir, percentage, which_vocab, vocab_path, batch_size, PADDING_IDX, shuffle
    )

    rnn_config = config['rnn_config']
    model = RNN(rnn_config).to(device)
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=10, min_lr=1e-4, verbose=True)

    vocab = make_vocab(config)
    """
    print("Starting pretraining...")
    for epoch in range(1, config['num_epoch'] + 1):
        model.train()
        train_loss = 0
        for data, lengths in tqdm(dataloader):
            lengths = [length - 1 for length in lengths]
            optimizer.zero_grad()
            data = data.to(device)
            preds = model(data, lengths)
            targets = pack_padded_sequence(data[:, 1:], lengths, batch_first=True, enforce_sorted=False).data
            loss = loss_function(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Pretraining Epoch {epoch}, Loss: {train_loss:.4f}")
        scheduler.step(train_loss)
        sampled_molecules = sample(model, vocab, batch_size=1024)
        num_valid, num_invalid = compute_valid_rate(sampled_molecules)
        valid_rate = num_valid / (num_valid + num_invalid)
        print(f"Pretraining Epoch {epoch}, Valid Rate: {valid_rate:.4f}")
        if epoch == config['num_epoch']:
            torch.save(model.state_dict(), pretrained_model_dir)
            
    """

    fine_tune_dir = "./dataset/finetune_cleaned.smi"
    fine_tune_dataloader, _ = dataloader_gen(
        fine_tune_dir, 1.0, which_vocab, vocab_path, batch_size, PADDING_IDX, shuffle=False
    )
    model.load_state_dict(torch.load(pretrained_model_dir))  # Load pretrained model
    fine_tune_model(model, fine_tune_dataloader, optimizer, scheduler, loss_function, vocab, fine_tune_epochs=200, out_dir=out_dir)
