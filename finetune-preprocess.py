import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize

RDLogger.DisableLog('rdApp.*')

class MolCleaner:
    def __init__(self):
        self.normalizer = MolStandardize.rdMolStandardize.Normalizer()
        self.choose_frag = MolStandardize.rdMolStandardize.LargestFragmentChooser()

    def process(self, mol):
        mol = Chem.MolFromSmiles(mol)
        if mol is not None:
            mol = self.normalizer.normalize(mol)
            mol = self.choose_frag.choose(mol)
            mol = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
            return mol
        else:
            return None

if __name__ == "__main__":

    in_path = "pretrain.csv"
    out_path = "dataset/pretrain_cleaned.smi"

    df = pd.read_csv(in_path)
    smiles = df['canonicalsmiles'].dropna().tolist()
    print("Number of SMILES before cleaning:", len(smiles))

    cleaner = MolCleaner()
    processed = []
    for mol in tqdm(smiles, desc="Processing SMILES"):
        mol = cleaner.process(mol)

        if mol is not None and 20 < len(mol) < 120:
            processed.append(mol)

    processed = set(processed)
    print("Number of SMILES after cleaning:", len(processed))

    with open(out_path, "w") as f:
        for mol in processed:
            f.write(mol + "\n")
    print(f"Cleaned SMILES saved to {out_path}")