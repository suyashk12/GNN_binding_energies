# Main Challenge Information

This sub-directory contains data and some initial code to help you get started. We utilize the Python `json` module to store the training data. 
You can access the data for the main challenge in `graph_data.json`.

We have the python module `helper.py` that helps to read and write dictionaries of graphs.

The JSON is structured in key-value pairs, where the key is the SMILES string of the molecule used for training. 
The value is a networkx graph, which includes attributes on nodes and edges. 
Please inspect, execute and read the output of `explain_graph_data.py` for a more detailed understanding of the data.

Additionally in `training_data_info.txt' we provide detials on the number of atom and orbital types contained in the dataset.

## Extra Challenge Information

The following data is not necessary for completing the main challenges.

The original data for the extra challenge is in `raw_data.json'. 
Please inspect, execute and read the output of `explain_raw_data.py' for the details of this challenge.

## Where to Start building your Model

We think that Graph Neural Networks may be an excellent choice to build these models.
If you are not familiar with Graph Neural Networks or AI for Molecular Science here are some resources to read to get started:
https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb
https://dmol.pub/dl/gnn.html

As a tech stack, you could use:
 - [PyTorch](https://github.com/pytorch) with [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) for graphs.
 - [Flax](https://github.com/google/flax)/[JAX](https://github.com/google/jax) with [jraph](https://github.com/google-deepmind/jraph) as graph library
 - [TensorFlow](https://github.com/tensorflow) with the [GNN sub-module](https://github.com/tensorflow/gnn) for graphs.

But these are just suggestions for inspiration. You are free to choose any model you like to solve this task.
We emphasize that graphs are permutation invariant, under permuting nodes and edges, so your result should be invariant under this input transformation.

Good luck!

