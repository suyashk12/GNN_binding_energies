Revised code for working paper on molecular binding energy prediction using GNNs

Collaborators: Adam Fouda, Ray Ding, Valay Agarwal, Joshua Zhoue, Bhavnesh Jangid, Jacob Wardzala, Rodrigo Ferreira, Suyash Kumar

Directory structure -

1. data

a. embeddings - contains SkipAtom embeddings under "embeddings". 30-dim version has "not-induced" and "induced" versions which correspond to pre- and post-training. The 200-dim version has one-hot encoding of orbitals at the end which you might need to remove.

b. graphs - includes cleaned graphs made by Josh (does not include the latest July 1 version), the raw data file, and JSON files for duplicated graphs for the core IP and Lawrence Berkeley data from Bhavnesh (see data processing code below)

c. lookups - isolated binding energy data from LBL from Rodrigo (isolated_binding_energies.csv) and Bhavnesh (lwnl.csv), and some basic periodic table data.

2. data processing

a. duplicate_graphs.ipynb - loads cleaned graph data and makes copies per target atom/orbital (because the current GNN architecture predicts only a global property)

b. isolated_be_compare.ipynb - compares the binding energy between isolated atoms in the cleaned graph data and LBL experimental data

c. lwnl_netx.ipynb - creates NetworkX graphs from LBL data prepared by Bhavnesh

3. gnn_training

a. gnn_lookup_message_passing.ipynb - trains a GNN (see architecture below) on the difference between experimental baseline prediction from LBL data and the binding energy in a molecular context.

4. GNN architecture.pdf - shows a previous version of the model architecture. Currently, the node features include the electronegativity score and the orbitals are represented with quantum numbers instead of one-hot encoding.
