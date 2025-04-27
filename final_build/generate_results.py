# Imports
import json
import networkx as nx
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
import pandas as pd

# File loads
input_file = 'graph_data.json' # Input for making predictions
duplicate_file = 'graph_data_duplicates.json' # File storing split copies of input graphs

# Supplemental data for making predictions - embeddings, lookup table, orbitals, and bonds
embedding_df = pd.read_csv("embeddings.csv")
lookup_df = pd.read_csv("lookup.csv")
embedded_atoms = list(embedding_df['Atoms'])
embed_dict = {}
for i in range(len(embedding_df)):
    vec = np.fromstring(embedding_df.loc[i,'Embeddings'].replace('\n','').strip('[]'), sep=' ')
    embed_dict[embedded_atoms[i]] = vec        
embed_dict['H'] = np.zeros(30)
atom_dict = {row['Element']: row['Z'] for _, row in lookup_df.iterrows()}

orb_list = ['1s',
         '2s',
         '2p',
         '2p3/2',
         '3s',
         '3p',
         '3p3/2',
         '3d',
         '3d5/2',
         '4s',
         '4p3/2',
         '4d',
         '4d5/2',
         '4f7/2',
         '5s',
         '5p3/2',
         '5d5/2']

bond_dict = {'SINGLE':1, 'DOUBLE':2, 'TRIPLE':3, 'NONE':0}

# Model file and result names
model_file = 'model.pth'
result_file = 'results.csv'


# Part 1 - create duplicate graphs per atom/orbital
def load_data_from_file(filename):
    with open(filename, "r") as file_handle:
        string_dict = json.load(file_handle)
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

def write_data_to_json_file(graph_dict, filename, **kwargs):
    with open(filename, "w") as file_handle:
        json_string = json.dumps(graph_dict, default=nx.node_link_data, **kwargs)
        file_handle.write(json_string)


def create_duplicates(input_file, output_file):
    # Load the input graph data
    train_data = load_data_from_file(input_file)

    # List to hold all graphs and names
    training_graphs = []
    training_graph_names = []

    # Process each graph in the training data
    for i in range(len(train_data)):  # For each graph in the training set
        mol = list(train_data.keys())[i]
        graph = train_data[mol]  # Graph to be duplicated
        
        # Get the target nodes (with orbitals, even if they are -1)
        target_node_indices = [n for n, v in graph.nodes(data=True) if v['orbitals']]

        # List to hold all duplicated graphs and names for this graph
        all_graphs = []
        all_names = []

        count = 0
        for n_i in target_node_indices:  # For each target node
            graph_i = graph.copy()

            # Print orbitals for inspection before filtering
            #print(f"Processing molecule {mol}: Node {n_i} orbitals before filtering: {graph_i.nodes[n_i]['orbitals']}")

            # Add 'NONE' bonds to all non-neighboring nodes of the target node
            for nb_i in nx.non_neighbors(graph_i, n_i):
                graph_i.add_edge(n_i, nb_i, bond_type='NONE')  # Add edge to target node
            
            # Create duplicates for each orbital of the target node, including those with -1
            for j in range(len(graph_i.nodes[n_i]['orbitals'])):
                graph_ij = graph_i.copy()

                # No skipping based on orbital value now
                graph_ij.nodes[n_i]['orbitals'] = [graph_ij.nodes[n_i]['orbitals'][j]]

                # Print orbitals after filtering for comparison
                #print(f"Processing molecule {mol}: Node {n_i} orbitals after filtering: {graph_ij.nodes[n_i]['orbitals']}")

                # Mark the target node and other nodes
                graph_ij.nodes[n_i]['target'] = True
                for n in graph_ij.nodes:
                    if n != n_i:
                        graph_ij.nodes[n]['target'] = False
                
                # Append the duplicated graph and its name
                all_graphs.append(graph_ij)
                name = f"{mol}_{n_i}_{j}"  # Name based on molecule, target node, and orbital index
                all_names.append(name)
                count += 1

        # Add the generated graphs and names to the overall lists
        training_graphs.extend(all_graphs)
        training_graph_names.extend(all_names)

        # Print how many duplicates were created for this graph
        #print(f"Graph {i} for molecule {mol} created {count} duplicates.")

    # Write the duplicates to a new JSON file
    training_data_dict = dict(zip(training_graph_names, training_graphs))
    write_data_to_json_file(training_data_dict, output_file, indent=2)

create_duplicates(input_file, duplicate_file)

# Part 2 - convert networkx graphs to PyG and perform predictions
class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )

        self.mlp_upd = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
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

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        self.pool = global_mean_pool

        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        h = self.lin_in(data.x) 

        for conv in self.convs:
            h = h + conv(h, data.edge_index, data.edge_attr)  
        h_graph = h.mean(dim=0, keepdim=True) 
        out = self.lin_pred(h_graph)  
        return out.view(-1)  
    
# Load graph data (duplicates)
graph_data = load_data_from_file(duplicate_file)

# Load the model, hard coded layer and embedding dimension with best parameters from hyperparameter tuning
model = MPNNModel(num_layers=6, emb_dim=94, in_dim=49, edge_dim=4, out_dim=1)
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
model.eval()  # Make sure the model is in evaluation mode

# Prepare a list for results
results = []

# Iterate through each graph in the dataset
for mol_name, graph in graph_data.items():
    
    # Get node attributes
    targets = list(nx.get_node_attributes(graph, "target").values())
    atom_idx = targets.index(True)
    atoms = list(nx.get_node_attributes(graph, "atom_type").values())
    orbitals = list(nx.get_node_attributes(graph, "orbitals").values())
    
    # Extract the target atom and orbital
    target_atom_idx = int(mol_name.split('_')[1])  # Get atom index from molecule name
    target_orbital = orbitals[target_atom_idx][0]  # Get the first orbital of the target atom
    
    if target_orbital == -1:
        # If orbital type is -1, prepend a predicted binding energy of -1
        results.append([mol_name.split('_')[0], atoms[target_atom_idx], target_orbital, -1])
        continue
    
    # Construct PyG graph
    x = np.zeros((len(targets), 49))
    
    # Set target atom
    x[target_atom_idx, 0] = 1  # target atom
    charge = list(nx.get_node_attributes(graph, "formal_charge").values())
    x[:, 1] = charge
    
    atom_embeds = np.array([embed_dict[a] for a in atoms])
    x[:, 2:32] = atom_embeds
    x[target_atom_idx, 32 + orb_list.index(target_orbital)] = 1  # orbital one-hot encoding for target
    
    x = torch.tensor(x, dtype=torch.float)
    
    # Index is Z-1 for lookup table
    energy = float(lookup_df.loc[atom_dict[atoms[target_atom_idx]]-1, target_orbital])
    
    edges = list(graph.edges)

    if len(edges) == 0:  # If there are no edges, add self-loops
        edge_index = torch.arange(len(targets), dtype=torch.long).view(1, -1).repeat(2, 1)
        edge_attr = torch.zeros(1, 4)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        bond_types = list(nx.get_edge_attributes(graph, "bond_type").values())
        edge_attr = np.zeros((4, len(bond_types)))
        for j, b in enumerate(bond_types):
            edge_attr[bond_dict[b], j] = 1
        edge_attr = torch.tensor(edge_attr, dtype=torch.long).t().contiguous()
    
    # PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=energy)
    
    # Model prediction
    with torch.no_grad():
        y_pred = model(data)
    
    # Prepend the predicted binding energy
    results.append([mol_name.split('_')[0], atoms[target_atom_idx], target_orbital, (y_pred.item() + data.y)])

results_df = pd.DataFrame(results, columns=['Molecule', 'Atom', 'Orbital', 'Binding Energy'])
results_df.to_csv(result_file, index=False)