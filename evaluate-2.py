from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from tqdm import tqdm

def compute_tanimoto(smiles1, smiles2, morgan_generator):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None
        fp1 = morgan_generator.GetFingerprint(mol1)
        fp2 = morgan_generator.GetFingerprint(mol2)
        tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
        return tanimoto
    except Exception:
        return None

if __name__ == "__main__":

    sampled_file = "sampled_molecules-reinforced-finetuned.out"
    finetune_file = "dataset/finetune_cleaned.smi"
    output_file = "tanimoto_analysis-bigsample-1.out"

    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2)

    with open(sampled_file, "r") as f:
        sampled_smiles = [line.strip() for line in f.readlines()]

    with open(finetune_file, "r") as f:
        finetune_smiles = [line.strip() for line in f.readlines()]

    with open(output_file, "w") as out_f:
        out_f.write("Highest Tanimoto - Matched SMILES - Test SMILES\n")

        for test_smiles in tqdm(sampled_smiles, desc="Processing sampled SMILES"):
            if Chem.MolFromSmiles(test_smiles) is None:
                continue

            highest_tanimoto = 0
            best_match_smiles = None

            for ref_smiles in finetune_smiles:
                if Chem.MolFromSmiles(ref_smiles) is None:
                    continue

                tanimoto = compute_tanimoto(test_smiles, ref_smiles, morgan_generator)
                if tanimoto is not None and tanimoto > highest_tanimoto:
                    highest_tanimoto = tanimoto
                    best_match_smiles = ref_smiles

            out_f.write(f"{highest_tanimoto:.4f} - {best_match_smiles} - {test_smiles}\n")

    print(f"Tanimoto analysis completed. Results saved to {output_file}.")