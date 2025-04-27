#@title [RUN] Import python modules
import os
import time
import random
import numpy as np

from scipy.stats import ortho_group

import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm, LeakyReLU, ReLU, BatchNorm1d, Module, Sequential

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.utils import remove_self_loops, to_dense_adj, dense_to_sparse
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_scatter import scatter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#@title [RUN] Set random seed for deterministic results

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed(1)

# For storing experimental results over the course of the practical
RESULTS = {}

# Convert networkx to PyG
import json
import networkx as nx
import copy

def load_data_from_file(filename):
    """
    Load a dictionary of graphs from JSON file.
    """
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
    return _load_data_from_string_dict(string_dict)

def load_data_from_string(json_string):
    """
    Load a dictionary of graphs from JSON string.
    """
    string_dict = json.loads(json_string)
    return _load_data_from_string_dict(string_dict)

def _load_data_from_string_dict(string_dict):
    result_dict = {}
    for key in string_dict:

        data = copy.deepcopy(string_dict[key])

        if 'edges' in data:
            data["links"] = data.pop("edges")

        graph = nx.node_link_graph(data)
        result_dict[key] = graph
    return result_dict

def write_data_to_json_string(graph_dict, **kwargs):
    """
    Write dictionary of graphs to JSON string.
    """
    json_string = json.dumps(graph_dict, default=nx.node_link_data, **kwargs)
    return json_string


def write_data_to_json_file(graph_dict, filename, **kwargs):
    """
    Write dictionary of graphs to JSON file.
    """
    with open(filename, "w") as file_handle:
        file_handle.write(write_data_to_json_string(graph_dict, **kwargs))

train_data = load_data_from_file("graph_data_duplicates_v2.json") # dictionary of SMILES and values are graphs

atom_dict = {
 'H':1,
 'He':2,
 'Li': 3,
 'B': 5,
 'C': 6,
 'N': 7,
 'O': 8,
 'F': 9,
 'Ne': 10,
 'Na': 11,
 'Mg': 12,
 'Al': 13,
 'Si': 14,
 'P': 15,
 'S': 16,
 'Cl': 17,
 'Ar': 18,
 'K': 19,
 'Ca': 20,
 'Ti': 22,
 'V': 23,
 'Cr': 24,
 'Mn': 25,
 'Fe': 26,
 'Co': 27,
 'Ni': 28,
 'Cu': 29,
 'Zn': 30,
 'Ga': 31,
 'Ge': 32,
 'As': 33,
 'Se': 34,
 'Br': 35,
 'Kr': 36,
 'Rb': 37,
 'Sr': 38,
 'Mo': 42,
 'Rh': 45,
 'Ag': 47,
 'Cd': 48,
 'In': 49,
 'Sn': 50,
 'Sb': 51,
 'Te': 52,
 'I': 53,
 'Xe': 54,
 'Cs': 55,
 'Ba': 56,
 'W': 74,
 'Re': 75,
 'Hg': 80,
 'Tl': 81,
 'Pb': 82,
 'Bi': 83,
 'U': 92
}

orb_dict = {
 '1s': [1, 0, 0],
 '2s': [2, 0, 0],
 '2p': [2, 1, 0],
 '2p3/2': [2, 1, 1.5],
    
 '3s': [3, 0, 0],
 '3p': [3, 1, 0],
 '3p3/2': [3, 1, 1.5],
 '3d': [3, 2, 0],
 '3d5/2': [3, 2, 2.5],
    
 '4s': [4, 0, 0],
 '4p3/2': [4, 1, 1.5],
 '4d': [4, 2, 0],
 '4d5/2': [4, 2, 2.5],
 '4f7/2': [4, 3, 3.5],
    
 '5s': [5, 0, 0],
 '5p3/2': [5, 1, 1.5],
 '5d5/2': [5, 2, 2.5],
}

orb_list = list(orb_dict.keys())

bond_dict = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'NONE':0}

import pandas as pd

# Lookup table
iso_be_df = pd.read_csv('Isolated_Energy_final.csv')

# Embeddings
embed_df = pd.read_csv('final_embedding_dim30_induced.csv')

# Embedding IDs
embedded_atoms = list(embed_df['Atoms'])

# Embedding vectors
embed_dict = {}

for i in range(len(embed_df)):
    vec = np.fromstring(embed_df.loc[i,'Embeddings'].replace('\n','').strip('[]'), sep=' ')
    embed_dict[embedded_atoms[i]] = vec
        
# Embedding doesn't have entry for H
embed_dict['H'] = np.zeros(30)

# Build dataset

dataset = []

for i, mol in enumerate(train_data):
    graph = train_data[mol]
    
    
    # Create node features (# atoms by target, atomic embedding, charge, one-hot encoding for orbital)
    targets = list(nx.get_node_attributes(graph, "target").values())
    x = np.zeros((len(targets), 49))

    # Indicate target atom
    atom_index = targets.index(True)
    x[atom_index, 0] = 1  
    
    # Indicate formal charges
    charge = list(nx.get_node_attributes(graph, "formal_charge").values())
    x[:, 1] = charge  
    
    # Indicate atom embeddings
    atom = list(nx.get_node_attributes(graph, "atom_type").values())
    embed_mat = np.array([embed_dict[a] for a in atom])
    x[:,2:32] = embed_mat
    
    # One hot encode orbital for target atom
    orb = list(nx.get_node_attributes(graph, "orbitals").values())
    x[atom_index, 32 + orb_list.index(orb[atom_index][0])] = 1 
    
    # Make PyG friendly
    x = torch.tensor(x, dtype=torch.float)
    
    # Target value
    energy = list(nx.get_node_attributes(graph, "binding_energies").values())
    y = torch.tensor([energy[atom_index][0]], dtype=torch.float)
    # Isolated binding energy from lookup table (for delta learning)
    be = float(iso_be_df.loc[atom_dict[atom[atom_index]]-1,orb[atom_index][0]]) 
    

    # Fix for edge_index with source and destination nodes
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()

    # Encode edge indices
    bond_types = list(nx.get_edge_attributes(graph, "bond_type").values())
    
    # Fix for edge_attr construction
    edge_attr = np.zeros((4, len(bond_types), ))
    
    for j,b in enumerate(bond_types):
        edge_attr[bond_dict[b], j] = 1
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.long).t().contiguous()

    # Then proceed with creating the Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y-be, name=mol, y0=be)
    dataset.append(data)

# Build training and testing set
from torch.utils.data import random_split

print(f"Total number of samples: {len(dataset)}.")

mols = np.array(list(train_data.keys()))
mols = np.array([mol.split('_')[0] for mol in mols]) # there are no underscores in the molecule names

# Get unique molecules and assign unique indices to each
unique_molecules, unique_index = np.unique(mols, return_inverse=True)

print(unique_index.max()) # goes from 0 to 859 for the 860 unique molecules

# Choose a random subset of test molecules
unseen_subset = np.random.choice(np.arange(0, 859), size=200, replace=False) # 100 random molecules

whitelist = ~np.in1d(unique_index, unseen_subset)
train_val_dataset = [dataset[i] for i in range(len(dataset)) if whitelist[i]]
test_dataset = [dataset[i] for i in range(len(dataset)) if not whitelist[i]] # Test dataset is the unseen subset

# Split datasets (our 3K subset)
# Split the dataset into train, validation, and test sets
train_size = int(0.75*len(train_val_dataset))
test_size = len(train_val_dataset)-train_size
split_seed = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, test_size], generator=split_seed)

print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

# Create dataloaders with batch size = 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Build message passing layer

# For message passing and update MLPs
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):

        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU(),   
            Linear(emb_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU()
          )

        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), 
            LayerNorm(emb_dim), ReLU()
          )

    def forward(self, h, edge_index, edge_attr):

        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):

        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        aggr_out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

        return aggr_out

    def update(self, aggr_out, h):

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')
    

class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=49, edge_dim=1, out_dim=1):

        super().__init__()

        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):

        h = self.lin_in(data.x) # (n, d_n) -> (n, d)

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr) # (n, d) -> (n, d)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)
    
#@title [RUN] Helper functions for managing experiments, training, and evaluating models.

def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        
        # Standard MSE loss
        mse_loss = F.mse_loss(y_pred, data.y)

        # L2 regularization (weight decay)
        l2_lambda = 1e-4  # regularization strength
        l2_reg = sum((param**2).sum() for param in model.parameters())     
        
        loss = F.mse_loss(y_pred, data.y) + l2_lambda * l2_reg
        
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)

    
def eval(model, loader, device):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            y_pred = model(data)
            error += ((y_pred - data.y)**2).sum().item()
    return error / len(loader.dataset)


def run_experiment(model, model_name, train_loader, val_loader, test_loader, n_epochs=100):
    
    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=10, min_lr=1e-7)

    early_stopping_patience = 10
    epochs_since_improvement = 0
    best_val_error = np.inf
    best_test_error = None
    best_epoch = -1
    
    print("\nStart training:")
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device)

        val_error = eval(model, val_loader, device)
        
        if val_error<best_val_error:
            best_val_error = val_error
            best_test_error = eval(model, test_loader, device)
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                  f'Val MSE: {val_error:.7f}, Test MSE: {best_test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((best_test_error, val_error, epoch, model_name))

        if epochs_since_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. "
                  f"No improvement in validation MSE for {early_stopping_patience} consecutive epochs.")
            break

    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation MSE: {best_val_error:.7f}, corresponding test MSE: {best_test_error:.7f}.")

    return best_val_error, best_test_error, train_time, perf_per_epoch

# Perform training loop
model = MPNNModel(num_layers=4, emb_dim=64, in_dim=49, edge_dim=4, out_dim=1)
model_name = type(model).__name__


best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    model,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    n_epochs=200
)

# Save model
torch.save(model.state_dict(), 'models_mlp_arch/model_0.pth')
DF_RESULTS = pd.DataFrame(perf_per_epoch, columns=["Test MSE", "Val MSE", "Epoch", "Model"])
DF_RESULTS.to_csv('models_mlp_arch/metrics_0.csv', index=False)

# Obtain predictions for test set
model.eval()

y_pred_list = []
y0_list = []
y_test_list = []

device = next(model.parameters()).device  # Automatically get model's device

for data in test_loader:
    data = data.to(device)
    with torch.no_grad():
        y_pred = model(data)
        
        y0_list += list(data.y0.cpu())
        y_pred_list += list((y_pred + data.y0).cpu())
        y_test_list += list((data.y + data.y0).cpu())

y0_arr = np.array(y0_list)
y_pred_arr = np.array(y_pred_list)
y_test_arr = np.array(y_test_list)

gnn_mae = np.mean(np.abs(y_pred_arr-y_test_arr))
lookup_mae = np.mean(np.abs(y0_arr-y_test_arr))

# Print performance
print('gnn_mae:' + str(gnn_mae) + ', lookup_mae:' + str(lookup_mae))