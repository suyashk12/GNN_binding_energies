# Notes on the PyTorch environment

When you run 
```
source setup_material_characterize.sh
```
A conda envrionemnt is created that uses the pytorch packages listed in gnnpytorch_requirements.txt

This environment enables teams to build and run their own gnn models from the tutorial below:

https://colab.research.google.com/github/chaitjo/geometric-gnn-dojo/blob/main/geometric_gnn_101.ipynb

tutorial_code_env_test.py loads all the nessesary packages to run the code from this tutorial and checks the gpu compatibilty

We think that Graph Neural Networks may be an excellent choice for this task
If you are not familiar with Graph Neural Networks or AI for Molecular Science this would be a great way started. 

However you are free to choose any model you like to solve this task.

This might require setting up you own environments.
We refer you to the docs for th list of possible tech stacks:
 - [PyTorch](https://github.com/pytorch) with [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) for graphs.
 - [Flax](https://github.com/google/flax)/[JAX](https://github.com/google/jax) with [jraph](https://github.com/google-deepmind/jraph) as graph library
 - [TensorFlow](https://github.com/tensorflow) with the [GNN sub-module](https://github.com/tensorflow/gnn) for graphs.


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


Good luck!

