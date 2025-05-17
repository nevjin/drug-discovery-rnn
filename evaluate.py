import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, DataStructs
from tqdm import tqdm

def lipinski_rule(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    return mw <= 500 and logp <= 5 and h_donors <= 5 and h_acceptors <= 10

def compute_tanimoto(smiles, reference_fp, morgan_generator):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = morgan_generator.GetFingerprint(mol)
        tanimoto = DataStructs.TanimotoSimilarity(fp, reference_fp)
        return tanimoto
    except Exception:
        return None

if __name__ == "__main__":

    input_file = "sampled_molecules-v4.out"

    reference_smiles = "CN1CCC[C@H]1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN([C@H](C6)CC#N)C(=O)C(=C)F"
    reference_mol = Chem.MolFromSmiles(reference_smiles)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    reference_fp = morgan_generator.GetFingerprint(reference_mol)

    highest_tanimoto = 0
    best_smiles = None
    lipinski_count = 0
    total_tanimoto = 0
    valid_smiles_count = 0

    with open(input_file, "r") as f:
        smiles_list = f.readlines()

    for smiles in tqdm(smiles_list):
        smiles = smiles.strip()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        if lipinski_rule(mol):
            lipinski_count += 1

        tanimoto = compute_tanimoto(smiles, reference_fp, morgan_generator)
        if tanimoto is not None:
            total_tanimoto += tanimoto
            valid_smiles_count += 1
            if tanimoto > highest_tanimoto:
                highest_tanimoto = tanimoto
                best_smiles = smiles

    avg_tanimoto = total_tanimoto / valid_smiles_count if valid_smiles_count > 0 else 0

    print(f"SMILES with the highest Tanimoto similarity: {best_smiles} (Similarity: {highest_tanimoto:.4f})")
    print(f"Number of SMILES that passed Lipinski's rule: {lipinski_count}")
    print(f"Average Tanimoto similarity of valid SMILES: {avg_tanimoto:.4f}")