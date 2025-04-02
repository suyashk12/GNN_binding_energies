import networkx as nx
import helper

train_data = helper.load_data_from_file("graph_data.json")

graph = train_data["C[CH2-].C[CH2-].[Zn+2]"]

print(" ")
print(f"We have {len(train_data.keys())} molecular graphs to train with.")
print("Please adhere to best practices during training.\n")

print("The networkx graphs in the graph_data.json file are lablled by SMILES strings")
print("For more details on SMILES, visit: https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system\n")
print("Below is an example of the graph nodes called by the SMILE string C[CH2-].C[CH2-].[Zn+2].\n")

print("Graph Nodes:")
for node, data in graph.nodes(data=True):
     print(f"Node {node}: {data}")

print("\nGraph Edges:")
for u, v, data in graph.edges(data=True):  
     print(f"Edge ({u}, {v}): {data}")
print(" ")

print("Let's examine the organization of the nodes by printing node 0:")
print("Remember, the index has no inherent meaning;")
print("your results should be consistent even if these indices are permuted.")
print("Permutation Invariance!\n")

print("'atom_type', 'formal_charge' and 'orbitals' are the attributes available for node featurization, 'binding_energies' are the node labels to predict:\n")
print("\t'atom_type' represents the atomic number, or the number of protons in an atom's nucleus.")
print("\t\tThis number differentiates elements, e.g., 6 for carbon and 1 for hydrogen.")
print("\t\tThis dataset contains 55 different atom types, use a one-hot encoding vector of length 55,")
print("\t\t... or try different atom representations available in the literaturw (SkipAtom, Mat2Vec ...)\n")

print("\t'formal_charge' is the integer charge of the atom in the molecule.\n")

print("\t'orbitals' is a vector of the orbitals types with binding energies values.")
print("\t\tThe length of this vector is same length as the binding energies vector.\n")

print("\t'binding_energies' is a vector of the output node labels to predict.\n")

print("The bnding energy data is sparse, notice that many of the orbitals and binding energies above are assigned -1 dummy values.\n")

print("Furthermore, some graph nodes will have many binding energy values.")
print("There are even some single atom graphs in the data.")
print("For example the Xe atom (SMILE string [Xe]) has many binding energies:\n")

atom_graph = train_data["[Xe]"]
print("Graph for [Xe]:")
print(atom_graph.nodes(data=True))
print(" ")

print("We suggest two ideas for handling the multidimensional nature of the outputs in the project presentation slides.\n") 

print("For the edges, the 'bond_type' feature is categorical and given as a string.\n") 
print("\tPossible types are SINGLE, DOUBLE, TRIPLE. use a one-hot encoding for the bond-types.")
print("\tIn the example above, Node 4 is not bonded to any other atoms.") 
print("\tTherefore you should turn this data into fully connected graphs and create another 'bond_type' catergory called NONE.\n")

print("The following google colab tutorial is great starting point for molecular graph neural networks in PyTorch:")
print("https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb")
print(" ")

