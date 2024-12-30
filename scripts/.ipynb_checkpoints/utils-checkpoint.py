import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data

class OmicsDataset(Dataset):
    def __init__(self, cell_graph_dict, drug_dict, data, pathway_tensor=None):
        self.cell_graph_dict = cell_graph_dict  
        self.drug_dict = drug_dict 
        self.cell_mapped_ids = {key: i for i, key in enumerate(self.cell_graph_dict.keys())}
        self.drug_mapped_ids = {key: i for i, key in enumerate(self.drug_dict.keys())}
        self.data = data 
        self.pathway_tensor = pathway_tensor 
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx=int(idx)
        instance = self.data.iloc[idx]
        cell_id = instance["SANGER_MODEL_ID"]
        drug_id = instance["DRUG_ID"]
        target = instance["LN_IC50"]

    
        cell_graph = self.cell_graph_dict.get(cell_id, None)
        if cell_graph is None:
            raise KeyError(f"Cell graph for cell_id {cell_id} not found in cell_graph_dict.")
        if not isinstance(cell_graph, Data):
            raise TypeError(f"Expected Data object for {cell_id}, got {type(cell_graph)}.")
    
        assert cell_graph is not None, f"cell_graph is None for cell_id {cell_id}"
    
        drug_tensor = self.drug_dict.get(drug_id, None)
        if drug_tensor is None:
            raise KeyError(f"Drug tensor for drug_id {drug_id} not found in drug_dict.")
        if not isinstance(drug_tensor, Tensor):
            drug_tensor = torch.tensor(drug_tensor, dtype=torch.float32)
    
        return (
            cell_graph, 
            drug_tensor,
            torch.tensor([target], dtype=torch.float32), 
            torch.tensor([self.cell_mapped_ids[cell_id]], dtype=torch.long), 
            torch.tensor([self.drug_mapped_ids[drug_id]], dtype=torch.long) 
        )

import rdkit
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = Tensor):
        
        self.R = R
        self.fp_kwargs = fp_kwargs
        self.fingerprint = fingerprint
        if fingerprint == "morgan":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(x, self.R, **fp_kwargs)
        elif fingerprint == "MACCS":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetMACCSKeysFingerprint(x, **fp_kwargs)
        elif fingerprint == "topological_torsion":
            self.f = lambda x: rdkit.Chem.rdMolDescriptors.GetTopologicalTorsionFingerprint(x, **fp_kwargs)
        self.transform = transform
    def __call__(self, smiles_list, drugs = None):
        drug_dict = {}
        if drugs is None:
            drugs = np.arange(len(smiles_list))
        for i in range(len(smiles_list)):
            try:
                smiles = smiles_list[i]
                molecule = AllChem.MolFromSmiles(smiles)
                feature_list = self.f(molecule)
                f = np.array(feature_list)
                if self.transform is not None:
                    f = self.transform(f)
                drug_dict[drugs[i]] = f
            except:
                drug_dict[drugs[i]] = None
        return drug_dict
    def __str__(self):
        
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"