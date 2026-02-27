from rdkit import Chem
import torch
import dgl
from typing import List

class Featurization_parameters:


    def __init__(self) -> None:
        self.MAX_ATOMIC_NUM = 54
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            'in_ring_smallest_size': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'lone_electron_pairs': [0, 1, 2, 3, 4],
            'H_bond_donor': [0, 1, 2, 3],
            'H_bond_acceptor': [0, 1, 2, 3],
        }



PARAMS = Featurization_parameters()





def atom_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """
    Builds a feature vector for each atom.
    return: A tensor containing the atom features.
    """
    features = []
    if atom:
        features = [
            onek_encoding_unk(atom.GetAtomicNum()-1, PARAMS.ATOM_FEATURES['atomic_num']) +
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            onek_encoding_unk(get_lone_electronpairs(atom), PARAMS.ATOM_FEATURES['lone_electron_pairs']) + \
            onek_encoding_unk(is_h_bond_donor(atom), PARAMS.ATOM_FEATURES['H_bond_donor']) + \
            onek_encoding_unk(is_h_bond_acceptor(atom), PARAMS.ATOM_FEATURES['H_bond_acceptor']) + \
            onek_encoding_unk(get_ring_size(atom), PARAMS.ATOM_FEATURES['in_ring_smallest_size']) + \
            [get_electronegativity(atom) * 0.1] + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]
        ]
    features = sum(features, [])
    return torch.tensor(features, dtype=torch.float)

def get_lone_electronpairs(atom: Chem.rdchem.Atom) -> int:

    symbol = atom.GetSymbol()
    if symbol == 'C' or symbol == 'H':
        return 0-atom.GetFormalCharge()
    elif symbol == 'S' or symbol == 'O':
        return 2-atom.GetFormalCharge()
    elif symbol == 'N' or symbol == 'P':
        return 1-atom.GetFormalCharge()
    elif symbol == 'F' or symbol == 'Cl' or symbol == 'Br' or symbol == 'I':
        return 3-atom.GetFormalCharge()
    else:
        return 0

def is_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
        return 1
    elif atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
        return 2
    elif atom.GetSymbol() == "F" and atom.GetTotalNumHs() > 0:
        return 3
    else:
        return 0

def is_h_bond_acceptor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and get_lone_electronpairs(atom) > 0:
        return 1
    elif atom.GetSymbol() == "O" and get_lone_electronpairs(atom) > 0:
        return 2
    elif atom.GetSymbol() == "F" and get_lone_electronpairs(atom) > 0:
        return 3
    else:
        return 0

def get_ring_size(atom: Chem.rdchem.Atom) -> int:
    if atom.IsInRing():
        for i in range(1, 8):
            if atom.IsInRingSize(i):
                return i
    else:
        return 0

def get_electronegativity(atom: Chem.rdchem.Atom) -> float:
    symbol = atom.GetSymbol()
    if symbol == "H":
        return 2.20
    elif symbol == "C":
        return 2.55
    elif symbol == "B":
        return 2.04
    elif symbol == "N":
        return 3.04
    elif symbol == "O":
        return 3.44
    elif symbol == "F":
        return 3.98
    elif symbol == "Al":
        return 1.61
    elif symbol == "Si":
        return 1.90
    elif symbol == "P":
        return 2.19
    elif symbol == "S":
        return 2.58
    elif symbol == "Cl":
        return 3.16
    elif symbol == "Br":
        return 2.96
    elif symbol == "I":
        return 2.66
    else:
        return 0

def bond_features(bond: Chem.rdchem.Bond) -> torch.Tensor:
    """
    Builds a feature vector for each bond and returns as a torch.Tensor.
    """
    bt = bond.GetBondType()
    fbond = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0
    ]
    stereo_features = onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    fbond += stereo_features
    fbond = [float(x) for x in fbond]

    return torch.tensor(fbond)


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding
from typing import List
from rdkit import Chem
import torch
import dgl

class Featurization_parameters:


    def __init__(self) -> None:
        self.MAX_ATOMIC_NUM = 54
        self.ATOM_FEATURES = {
            'atomic_num': list(range(self.MAX_ATOMIC_NUM)),
            'degree': [0, 1, 2, 3, 4, 5],
            'formal_charge': [-1, -2, 1, 2, 0],
            'num_Hs': [0, 1, 2, 3, 4],
            'hybridization': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ],
            'in_ring_smallest_size': [0, 1, 2, 3, 4, 5, 6, 7, 8],
            'lone_electron_pairs': [0, 1, 2, 3, 4],
            'H_bond_donor': [0, 1, 2, 3],
            'H_bond_acceptor': [0, 1, 2, 3],
        }



PARAMS = Featurization_parameters()





def atom_features(atom: Chem.rdchem.Atom) -> torch.Tensor:
    """
    Builds a feature vector for each atom.
    return: A tensor containing the atom features.
    """
    features = []
    if atom:
        features = [
            onek_encoding_unk(atom.GetAtomicNum()-1, PARAMS.ATOM_FEATURES['atomic_num']) +
            onek_encoding_unk(atom.GetTotalDegree(), PARAMS.ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), PARAMS.ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), PARAMS.ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), PARAMS.ATOM_FEATURES['hybridization']) + \
            onek_encoding_unk(get_lone_electronpairs(atom), PARAMS.ATOM_FEATURES['lone_electron_pairs']) + \
            onek_encoding_unk(is_h_bond_donor(atom), PARAMS.ATOM_FEATURES['H_bond_donor']) + \
            onek_encoding_unk(is_h_bond_acceptor(atom), PARAMS.ATOM_FEATURES['H_bond_acceptor']) + \
            onek_encoding_unk(get_ring_size(atom), PARAMS.ATOM_FEATURES['in_ring_smallest_size']) + \
            [get_electronegativity(atom) * 0.1] + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]
        ]
    features = sum(features, [])
    return torch.tensor(features, dtype=torch.float)

def get_lone_electronpairs(atom: Chem.rdchem.Atom) -> int:

    symbol = atom.GetSymbol()
    if symbol == 'C' or symbol == 'H':
        return 0-atom.GetFormalCharge()
    elif symbol == 'S' or symbol == 'O':
        return 2-atom.GetFormalCharge()
    elif symbol == 'N' or symbol == 'P':
        return 1-atom.GetFormalCharge()
    elif symbol == 'F' or symbol == 'Cl' or symbol == 'Br' or symbol == 'I':
        return 3-atom.GetFormalCharge()
    else:
        return 0

def is_h_bond_donor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0:
        return 1
    elif atom.GetSymbol() == "O" and atom.GetTotalNumHs() > 0:
        return 2
    elif atom.GetSymbol() == "F" and atom.GetTotalNumHs() > 0:
        return 3
    else:
        return 0

def is_h_bond_acceptor(atom: Chem.rdchem.Atom) -> int:
    if atom.GetSymbol() == "N" and get_lone_electronpairs(atom) > 0:
        return 1
    elif atom.GetSymbol() == "O" and get_lone_electronpairs(atom) > 0:
        return 2
    elif atom.GetSymbol() == "F" and get_lone_electronpairs(atom) > 0:
        return 3
    else:
        return 0

def get_ring_size(atom: Chem.rdchem.Atom) -> int:
    if atom.IsInRing():
        for i in range(1, 8):
            if atom.IsInRingSize(i):
                return i
    else:
        return 0

def get_electronegativity(atom: Chem.rdchem.Atom) -> float:
    symbol = atom.GetSymbol()
    if symbol == "H":
        return 2.20
    elif symbol == "C":
        return 2.55
    elif symbol == "B":
        return 2.04
    elif symbol == "N":
        return 3.04
    elif symbol == "O":
        return 3.44
    elif symbol == "F":
        return 3.98
    elif symbol == "Al":
        return 1.61
    elif symbol == "Si":
        return 1.90
    elif symbol == "P":
        return 2.19
    elif symbol == "S":
        return 2.58
    elif symbol == "Cl":
        return 3.16
    elif symbol == "Br":
        return 2.96
    elif symbol == "I":
        return 2.66
    else:
        return 0

def bond_features(bond: Chem.rdchem.Bond) -> torch.Tensor:
    """
    Builds a feature vector for each bond and returns as a torch.Tensor.
    """
    bt = bond.GetBondType()
    fbond = [
        1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.IsInRing() else 0.0
    ]
    stereo_features = onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    fbond += stereo_features
    fbond = [float(x) for x in fbond]

    return torch.tensor(fbond)


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    num_atoms = mol.GetNumAtoms()
    src, dst, edge_feats = [], [], []


    atom_feats = [atom_features(mol.GetAtomWithIdx(i)) for i in range(num_atoms)]

   
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        src.extend([u, v])
        dst.extend([v, u])
        edge_feats.extend([feat, feat])

    
    g = dgl.graph((src, dst), num_nodes=num_atoms)
    g.ndata['feat'] = torch.stack(atom_feats)
    g.edata['feat'] = torch.stack(edge_feats)

    return g

def smiles_to_graphs(smiles):
    graph = smiles_to_graph(smiles)
    return graph

class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, extra_features):
        self.smiles_list = smiles_list
        self.extra_features = extra_features

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        extra_features = self.extra_features[idx]

        
        graph = smiles_to_graphs(smiles)

        return graph, extra_features  
    def __len__(self):
        return len(self.smiles_list)

def collate_fn(samples):
    graphs, extra_features = zip(*samples)
    batched_graph = dgl.batch(graphs)
    extra_features = torch.stack(extra_features)

    return batched_graph, extra_features  
