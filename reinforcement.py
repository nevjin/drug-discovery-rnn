import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
import numpy as np
from tqdm import tqdm

from dataloader import SELFIEVocab, RegExVocab, CharVocab
from model import RNN

class ReinforcementLearning:
    def __init__(self, config_path, reference_smiles, device=None):

        with open(config_path, 'r') as f:
            self.config = yaml.full_load(f)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        which_vocab = self.config["which_vocab"]
        vocab_path = self.config["vocab_path"]

        if which_vocab == "selfies":
            self.vocab = SELFIEVocab(vocab_path)
        elif which_vocab == "regex":
            self.vocab = RegExVocab(vocab_path)
        elif which_vocab == "char":
            self.vocab = CharVocab(vocab_path)
        else:
            raise ValueError("Wrong vocab name in configuration!")

        rnn_config = self.config['rnn_config']
        self.model = RNN(rnn_config).to(self.device)
        self.model.load_state_dict(torch.load(
            self.config['out_dir'] + 'reinforced_finetuned_model_epoch50.pt',
            map_location=self.device))

        self.model.train()

        self.reference_smiles = reference_smiles
        self.reference_mol = Chem.MolFromSmiles(reference_smiles)
        self.reference_morgan = AllChem.GetMorganFingerprintAsBitVect(self.reference_mol, 2)

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def compute_reward(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return -1.0

            warhead_mol = Chem.MolFromSmarts("C=CC(=O)N")
            has_warhead = mol.HasSubstructMatch(warhead_mol)

            mol_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
            similarity = TanimotoSimilarity(self.reference_morgan, mol_morgan)

            mol_weight = Descriptors.ExactMolWt(mol)
            logp = Crippen.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)

            lipinski_score = sum([
                mol_weight < 500,
                logp < 5,
                h_donors <= 5,
                h_acceptors <= 10
            ])

            reward = 0
            if has_warhead:
                reward += 2.0
            reward += similarity * 1.5
            if lipinski_score >= 3:
                reward += 0.5

            return reward

        except Exception as e:
            return -1.0

    def policy_gradient_loss(self, log_probs, rewards):

        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-10)

        loss = -torch.mean(log_probs * rewards)

        return loss

    def reinforce(self, num_epochs=10, batch_size=128):
        best_reward = float('-inf')
        for epoch in range(num_epochs):

            sampled_ints = self.model.sample(
                batch_size=batch_size,
                vocab=self.vocab,
                device=self.device
            )

            sampled_smiles = []
            sampled_ints_list = sampled_ints.tolist()
            for ints in sampled_ints_list:
                molecule = []
                for x in ints:
                    if self.vocab.int2tocken[x] == '<eos>':
                        break
                    molecule.append(self.vocab.int2tocken[x])
                smiles = "".join(molecule)

                if self.vocab.name == 'selfies':
                    smiles = sf.decoder(smiles)

                sampled_smiles.append(smiles)

            rewards = [self.compute_reward(smiles) for smiles in sampled_smiles]

            sampled_ints, log_probs = self.model.sample(
                batch_size=batch_size,
                vocab=self.vocab,
                device=self.device,
                return_log_probs=True
            )

            loss = self.policy_gradient_loss(log_probs, rewards)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            avg_reward = np.mean(rewards)
            print(f"Epoch {epoch+1}: Avg Reward = {avg_reward}, Loss = {loss.item()}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(self.model.state_dict(),
                           f"{self.config['out_dir']}/best_reinforced_model.pt")
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(),
                           f"{self.config['out_dir']}/v2-reinforced-epoch{epoch+1}.pt")

        return self.model

def main():
    reference_smiles = "CN1CCC[C@H]1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN([C@H](C6)CC#N)C(=O)C(=C)F"

    config_path = "train.yaml"

    rl = ReinforcementLearning(config_path, reference_smiles)
    reinforced_model = rl.reinforce(num_epochs=10, batch_size=512)

if __name__ == "__main__":
    main()