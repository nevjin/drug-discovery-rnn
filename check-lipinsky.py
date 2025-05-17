from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors

def lipinski_rules(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "Invalid SMILES string"

    mol_weight = Descriptors.MolWt(mol)
    rule_1 = mol_weight <= 500

    logP = Crippen.MolLogP(mol)
    rule_2 = logP <= 5

    h_donors = Descriptors.NumHDonors(mol)
    rule_3 = h_donors <= 5

    h_acceptors = Descriptors.NumHAcceptors(mol)
    rule_4 = h_acceptors <= 10

    rules_followed = sum([rule_1, rule_2, rule_3, rule_4])

    return f"The molecule follows {rules_followed} out of 4 Lipinski's rules."

smiles_input = input("Enter a SMILES string: ")
result = lipinski_rules(smiles_input)
print(result)