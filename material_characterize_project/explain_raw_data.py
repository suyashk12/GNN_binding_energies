from rdkit import Chem
from rdkit.Chem import Draw
import helper
import json
import networkx as nx
from networkx.readwrite import node_link_graph


# load graph data
graph_data = helper.load_data_from_file("graph_data.json")

#example smile string from data
smiles = "C(#CC(F)(F)F)C(F)(F)F"

#select graph for example
graph = graph_data[smiles]

print("\nThe extra challenge involves finding a way to make more precise model predictions.")
print("This requires improving how the data is extracted from the original database to the graphs.\n")

print("Here are some example graph nodes from the main challenge:\n")
print("Graph Nodes:")
for node, data in graph.nodes(data=True):
     print(f"Node {node}: {data}")
print(" ")

print("You may notice that all the C nodes have the same binding energy value.")
print("When in reality the binding energy depends on the neighboring atom types, i.e. the local bonding environemnt.\n") 
print("The values in the graphs are infact an average of the C binding energies in different environemnts.\n") 
print("The specific binding energies are provided in the orginal data.")
print("However, assigning the original values to the correct atom indexes in the graphs is not trivial.\n")

print("The raw_data.json file has the orginal values from the online database:\n")

# load raw data
with open("raw_data.json", "r") as f:
    raw_data = json.load(f)

raw_data_example = raw_data[smiles]

print("Each item contains a dictionary, keyed by the SMILES string,") 
print("each value is another dictionary with the following structure:")
print(" {")
print("   'chemical_formula': [list of chemical formulas],")
print("   'orbitals': [list of orbitals],")
print("   'binding_energies': [list of binding energies],")
print("   'references': [list of reference comments or data]")
print(" }")

print("\nIf we print the items (except references) of the dictionary as columns we can see how the binding energies are assigned:\n")
items_except_ref = list(raw_data_example.values())[:-1]
for row in zip(*items_except_ref):
    print(*row)

print("\nThe * in the chemical formulas indicates which atom the binding energy value is for.") 
print("The order of the atoms in the chemical formauls implicitly contains the bonding information of the molecule,")
print("(see 'Condensed Formula' in https://en.wikipedia.org/wiki/Chemical_formula).\n") 

print("Making the correct assignments from the chemical formulas requires visual inspection of the molecules structure.")
print("rdkit can produce a 2D image of the moelcule from the smile string, which is written to example_mol.png by this code.\n")

# Convert smikes to rdkit mol and generate 2D image for reference
mol    = Chem.MolFromSmiles(smiles)
Chem.rdDepictor.Compute2DCoords(mol)
img = Draw.MolToImage(mol)
img.save("example_mol.png")

print("By looking at the chemical formuals and the structure:")
print("\tThe CF3C*C*CF3 binding energies correspond the the carbons in the triple bond in the center of the molecule")
print("\tThe C*F3CCC*F3 binding energies correspond to the carbons surrounded by fluorines.\n") 

print("Notice that original data has two values for each carbon type and for the fluorines (which are all in identical environments).")
print("This is becasue the dataset set contians values from mutliple references, given in the 'references' dict. key.")
print("The differences between the references are minor and either ref can be chosen.\n") 

print("The challenge is to find a way to make the correct assignments for the graphs across the dataset.")
print("One possible route could be to find cases where there is only one binding energy for an atom type in the molecule.") 
print("And use this information in a lookup table or to possibly train a model to learn how the nearest bond neighbors affects the values.\n")

print("This is a hard task and the original data contains many inconsitencies,") 
print("However completing this could enable your GNN models to achieve publishable results.\n")

print("To asses this challenge we have the correct graph data, manually assigned for a set of organic molecules (only contain H, C, N, O, F),")
print("If you reach this stage, please contact your mentor for the assessment data.\n")

