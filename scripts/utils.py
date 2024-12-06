from torch.utils.data import Dataset
from torch import Tensor
import numpy as np


class OmicsDataset(Dataset):
    def __init__(self, cell_graph_dict, drug_dict, data):
        """
        Dataset class for handling cell graph and drug data.

        Args:
            cell_graph_dict (dict): Dictionary mapping cell line IDs to graph Data objects.
            drug_dict (dict): Dictionary mapping drug IDs to their featurized tensors.
            data (DataFrame): DataFrame containing `SANGER_MODEL_ID`, `DRUG_ID`, and target (e.g., IC50).
        """
        self.cell_graph_dict = cell_graph_dict
        self.drug_dict = drug_dict
        self.cell_mapped_ids = {key: i for i, key in enumerate(self.cell_graph_dict.keys())}
        self.drug_mapped_ids = {key: i for i, key in enumerate(self.drug_dict.keys())}
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an instance from the dataset.

        Args:
            idx (int): Index of the instance.

        Returns:
            tuple: (cell_graph, drug_tensor, target, cell_line_id, drug_id)
        """
        instance = self.data.iloc[idx]
        cell_id = instance["SANGER_MODEL_ID"]
        drug_id = instance["DRUG_ID"]
        target = instance["target"]  # Adjust column name to match your target column

        # Return graph, drug tensor, and other identifiers
        return (
            self.cell_graph_dict[cell_id],  # PyTorch Geometric Data object
            self.drug_dict[drug_id],       # Drug tensor
            Tensor([target]),              # Target value (e.g., IC50)
            Tensor([self.cell_mapped_ids[cell_id]]),  # Cell line ID as Tensor
            Tensor([self.drug_mapped_ids[drug_id]])   # Drug ID as Tensor
        )

    
    
import rdkit
from rdkit.Chem import AllChem
class FingerprintFeaturizer():
    def __init__(self,
                 fingerprint = "morgan",
                 R=2, 
                 fp_kwargs = {},
                 transform = Tensor):
        """
        Get a fingerprint from a list of molecules.
        Available fingerprints: MACCS, morgan, topological_torsion
        R is only used for morgan fingerprint.
        fp_kwards passes the arguments to the rdkit fingerprint functions:
        GetMorganFingerprintAsBitVect, GetMACCSKeysFingerprint, GetTopologicalTorsionFingerprint
        """
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
        """
        returns a description of the featurization
        """
        return f"{self.fingerprint}Fingerprint_R{self.R}_{str(self.fp_kwargs)}"