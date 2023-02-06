import torch
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import substra
import inquirer
from substra import Client
from substra.sdk.schemas import Permissions
from substra.sdk.schemas import DataSampleSpec
from substra.sdk.schemas import DatasetSpec
from substrafl.strategies import FedAvg, SingleOrganization, Scaffold, __all__ as strategies
from substrafl.model_loading import load_algo
from substrafl.model_loading import download_algo_files
from substrafl.experiment import execute_experiment
from substrafl.evaluation_strategy import EvaluationStrategy
from substrafl.nodes import TestDataNode
from substrafl.nodes import AggregationNode
from substrafl.nodes import TrainDataNode
from substrafl.algorithms.pytorch import TorchFedAvgAlgo, TorchScaffoldAlgo, TorchSingleOrganizationAlgo
from substrafl.index_generator import NpIndexGenerator
from substrafl.remote.register import add_metric
from substrafl.dependency import Dependency
from flamby.datasets import fed_heart_disease


questions = [
    inquirer.List('model_type',
                  message="What model would you like to use?",
                  choices=fed_heart_disease.models),
    inquirer.List('optimiser',
                  message="What optimiser would you like to use?",
                  choices=fed_heart_disease.optimizers),
    inquirer.List('federated_stat',
                  message="What federated strategy would you like to use?",
                  choices=strategies)
]


N_CLIENTS = fed_heart_disease.NUM_CLIENTS
MODE = substra.BackendType.LOCAL_DOCKER

# Create the substra clients
clients = [Client(backend_type=MODE) for _ in range(N_CLIENTS)]

# Store clients in a dict with their org id as key
data_provider_clients = {client.organization_info(
).organization_id: client for client in clients}

# Create the algo provider client
algo_provider_client = Client(backend_type=MODE)

# Store their IDs
DATA_PROVIDER_ORGS_ID = list(data_provider_clients.keys())

# The org id on which your computation tasks are registered
ALGO_ORG_ID = algo_provider_client.organization_info().organization_id

# %%
# Dataset registration
# ====================
#
# A :ref:`documentation/concepts:Dataset` is composed of an **opener**, which is a Python script that can load
# the data from the files in memory and a description markdown file.
# The :ref:`documentation/concepts:Dataset` object itself does not contain the data. The proper asset that contains the
# data is the **datasample asset**.
#
# A **datasample** contains a local path to the data. A datasample can be linked to a dataset in order to add data to a
# dataset.
#
# Data privacy is a key concept for Federated Learning experiments. That is why we set
# :ref:`documentation/concepts:Permissions` for each :ref:`documentation/concepts:Assets` to define which organization
# can use them.
#
# Note that metadata, for instance: assets' creation date, assets owner, are visible by all the organizations of a
# network.

# Path to empty data samples
assets_directory = pathlib.Path.cwd() / "assets"
empty_path = assets_directory / "empty_datasamples"

# Permissions for the data samples
permissions_dataset = Permissions(public=False, authorized_ids=[ALGO_ORG_ID])

# Train and test datasets keys
train_dataset_keys = {}
test_dataset_keys = {}
train_datasample_keys = {}
test_datasample_keys = {}


for i, org_id in enumerate(DATA_PROVIDER_ORGS_ID):

    client = data_provider_clients[org_id]

    # DatasetSpec is the specification of a dataset. It makes sure every field
    # is well defined, and that our dataset is ready to be registered.
    # The real dataset object is created in the add_dataset method.

    dataset = DatasetSpec(
        name="FLamby",
        type="torchDataset",
        data_opener=assets_directory / "dataset" / f"opener_train_{org_id}.py",
        description=assets_directory / "dataset" / "description.md",
        permissions=permissions_dataset,
        logs_permission=permissions_dataset,
    )

    # Add the dataset to the client to provide access to the opener in each organization.
    train_dataset_key = client.add_dataset(dataset)
    assert train_dataset_key, "Missing data manager key"

    train_dataset_keys[org_id] = train_dataset_key

    # Add the training data on each organization.
    data_sample = DataSampleSpec(
        data_manager_keys=[train_dataset_key],
        test_only=False,
        path=empty_path,
    )
    train_datasample_key = client.add_data_sample(
        data_sample,
        local=True,
    )
    train_datasample_keys[org_id] = train_datasample_key

    # Add the testing data.
    test_dataset_key = client.add_dataset(
        DatasetSpec(
            name="FLamby",
            type="torchDataset",
            data_opener=assets_directory / "dataset" / f"opener_test_{org_id}.py",
            description=assets_directory / "dataset" / "description.md",
            permissions=permissions_dataset,
            logs_permission=permissions_dataset,
        )
    )
    assert test_dataset_key, "Missing data manager key"
    test_dataset_keys[org_id] = test_dataset_key

    data_sample = DataSampleSpec(
        data_manager_keys=[test_dataset_key],
        test_only=True,
        path=empty_path,
    )
    test_datasample_key = client.add_data_sample(
        data_sample,
        local=True,
    )
    test_datasample_keys[org_id] = test_datasample_key


# %%
# Metric registration
# ===================
#
# A metric is a function used to compute the score of predictions on one or several
# **datasamples**.
#
# To add a metric, you need to define a function that computes and return a performance
# from the datasamples (as returned by the opener) and the predictions_path (to be loaded within the function).
#
# When using a Torch SubstraFL algorithm, the predictions are saved in the `predict` function under the numpy format
# so that you can simply load them using `np.load`.
#
# After defining the metrics dependencies and permissions, we use the `add_metric` function to register the metric.
# This metric will be used on the test datasamples to evaluate the model performances.
def fed_heart_disease_metric(datasamples, predictions_path):

    config = datasamples
    dataset = fed_heart_disease.FedHeartDisease(**config)
    dataloader = data.DataLoader(dataset, batch_size=len(dataset))

    y_true = next(iter(dataloader))[1]
    y_pred = np.load(predictions_path)

    return float(fed_heart_disease.metric_fed(y_true, y_pred))


# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
# The local dependencies are local packages to be installed using the command `pip install -e .`.
# Flamby is a local dependency. We put as argument the path to the `setup.py` file.
metric_deps = Dependency(pypi_dependencies=["torch==1.11.0", "numpy==1.23.1"],
                         # Flamby dependency
                         local_dependencies=[pathlib.Path.cwd().parent.parent],
                         )
permissions_metric = Permissions(
    public=False, authorized_ids=DATA_PROVIDER_ORGS_ID + [ALGO_ORG_ID])

metric_key = add_metric(
    client=algo_provider_client,
    metric_function=fed_heart_disease_metric,
    permissions=permissions_metric,
    dependencies=metric_deps,
)

# %%
# Specify the machine learning components
# ***************************************
#
# This section uses the PyTorch based SubstraFL API to simplify the machine learning components definition.
# However, SubstraFL is compatible with any machine learning framework.
#
# In this section, you will:
#
# - register a model and its dependencies
# - specify the federated learning strategy
# - specify the organizations where to train and where to aggregate
# - specify the organizations where to test the models
# - actually run the computations


# %%
# Model definition
# ================
# To load a model replace model and remove selection statements with the following:
# model = torch.load('models/model_pooled.pt', map_location=device)
#
answer = inquirer.prompt(questions)

SEED = 42

model = None
optimizer = None
criterion = fed_heart_disease.BaselineLoss()

if answer["model_type"] == "MLP":
    print("Using MLP model")
    model = fed_heart_disease.MLP()
elif answer["model_type"] == "Baseline":
    print("Using MLP model")
    model = fed_heart_disease.Baseline()


if answer["optimiser"] == "SDG":
    print("Using SGD optimiser")
    optimizer = fed_heart_disease.SGDOptimizer(
        model.parameters(), lr=fed_heart_disease.LR)
elif answer["optimiser"] == "Adam":
    print("Using Adam optimiser")
    optimizer = fed_heart_disease.AdamOptimizer(
        model.parameters(), lr=fed_heart_disease.LR)

use_gpu = torch.has_mps
device = "cpu"

if use_gpu:
    device = torch.device("mps")
    model = model.to(device)

# %%
# Specifying on how much data to train
# ====================================
#
# To specify on how much data to train at each round, we use the `index_generator` object.
# We specify the batch size and the number of batches to consider for each round (called `num_updates`).
# See :ref:`substrafl_doc/substrafl_overview:Index Generator` for more details.


# Number of model update between each FL strategy aggregation.
NUM_UPDATES = 16

# Number of samples per update.
BATCH_SIZE = fed_heart_disease.BATCH_SIZE

index_generator = NpIndexGenerator(
    batch_size=BATCH_SIZE,
    num_updates=NUM_UPDATES,
)

# %%
# Torch Dataset definition
# ==========================
#
# This torch Dataset is used to preprocess the data using the `__getitem__` function.
#
# This torch Dataset needs to have a specific `__init__` signature, that must contain (self, datasamples, is_inference).
#
# The `__getitem__` function is expected to return (inputs, outputs) if `is_inference` is `False`, else only the inputs.
# This behavior can be changed by re-writing the `_local_train` or `predict` methods.


class TorchDataset(fed_heart_disease.FedHeartDisease):
    def __init__(self, datasamples, is_inference):
        config = datasamples
        super().__init__(**config)


# %%
# Federated Learning strategies
# =============================
#
# A FL strategy specifies how to train a model on distributed data.
# The most well known strategy is the Federated Averaging strategy: train locally a model on every organization,
# then aggregate the weight updates from every organization, and then apply locally at each organization the averaged
# updates.
strategy = None
TORCH_ALGO = None

if answer["federated_stat"] == "FedAvg":
    print("Using FedAvg")
    strategy = FedAvg()
    TORCH_ALGO = TorchFedAvgAlgo

elif answer["federated_stat"] == "SingleOrganization":
    print("Using SingleOrganization")
    strategy = SingleOrganization()
    TORCH_ALGO = TorchSingleOrganizationAlgo

elif answer["federated_stat"] == "Scaffold":
    print("Using Scaffold")
    strategy = Scaffold()
    TORCH_ALGO = TorchScaffoldAlgo

elif answer["federated_stat"] == "NewtonRaphson" or answer["federated_stat"] == "Strategy":
    print("NewtonRaphson and Strategy algorithms are not yet supported, using FedAvg")
    strategy = FedAvg()
    TORCH_ALGO = TorchFedAvgAlgo

# %%
# SubstraFL algo definition
# ==========================
#
# A SubstraFL Algo gathers all the elements that we defined that run locally in each organization.
# This is the only SubstraFL object that is framework specific (here PyTorch specific).
#
# The `TorchDataset` is passed **as a class** to the `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_.
# Indeed, this `TorchDataset` will be instantiated directly on the data provider organization.


class MyAlgo(TORCH_ALGO):
    def __init__(self):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            index_generator=index_generator,
            dataset=TorchDataset,
            seed=SEED,
        )

    def _local_predict(self, predict_dataset: torch.utils.data.Dataset, predictions_path):

        batch_size = self._index_generator.batch_size
        predict_loader = torch.utils.data.DataLoader(
            predict_dataset, batch_size=batch_size)

        self._model.eval()

        # The output dimension of the model is of size (1,)
        predictions = torch.zeros((len(predict_dataset), 1))

        with torch.inference_mode():
            for i, (x, _) in enumerate(predict_loader):
                x = x.to(self._device)
                predictions[i * batch_size: (i+1) * batch_size] = self._model(x)

        predictions = predictions.cpu().detach()
        self._save_predictions(predictions, predictions_path)


# %%
# Where to train where to aggregate
# =================================
#
# We specify on which data we want to train our model, using the :ref:`substrafl_doc/api/nodes:TrainDataNode` objets.
# Here we train on the two datasets that we have registered earlier.
#
# The :ref:`substrafl_doc/api/nodes:AggregationNode` specifies the organization on which the aggregation operation
# will be computed.
aggregation_node = AggregationNode(ALGO_ORG_ID)

train_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Train Data Node (or training task) and save it in a list
    train_data_node = TrainDataNode(
        organization_id=org_id,
        data_manager_key=train_dataset_keys[org_id],
        data_sample_keys=[train_datasample_keys[org_id]],
    )
    train_data_nodes.append(train_data_node)

# %%
# Where and when to test
# ======================
#
# With the same logic as the train nodes, we create :ref:`substrafl_doc/api/nodes:TestDataNode` to specify on which
# data we want to test our model.
#
# The :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy` defines where and at which frequency we
# evaluate the model, using the given metric(s) that you registered in a previous section.

test_data_nodes = list()

for org_id in DATA_PROVIDER_ORGS_ID:

    # Create the Test Data Node (or testing task) and save it in a list
    test_data_node = TestDataNode(
        organization_id=org_id,
        data_manager_key=test_dataset_keys[org_id],
        test_data_sample_keys=[test_datasample_keys[org_id]],
        metric_keys=[metric_key],
    )
    test_data_nodes.append(test_data_node)

# Test at the end of every round
my_eval_strategy = EvaluationStrategy(test_data_nodes=test_data_nodes, rounds=1)

# %%
# Running the experiment
# **********************
#
# We now have all the necessary objects to launch our experiment. Please see a summary below of all the objects we created so far:
#
# - A :ref:`documentation/references/sdk:Client` to add or retrieve the assets of our experiment, using their keys to
#   identify them.
# - An `Torch algorithm <substrafl_doc/api/algorithms:Torch Algorithms>`_ to define the training parameters *(optimizer, train
#   function, predict function, etc...)*.
# - A `Federated Strategy <substrafl_doc/api/strategies:Strategies>`_, to specify how to train the model on
#   distributed data.
# - `Train data nodes <substrafl_doc/api/nodes:TrainDataNode>`_ to indicate on which data to train.
# - An :ref:`substrafl_doc/api/evaluation_strategy:Evaluation Strategy`, to define where and at which frequency we
#   evaluate the model.
# - An :ref:`substrafl_doc/api/nodes:AggregationNode`, to specify the organization on which the aggregation operation
#   will be computed.
# - The **number of rounds**, a round being defined by a local training step followed by an aggregation operation.
# - An **experiment folder** to save a summary of the operation made.
# - The :ref:`substrafl_doc/api/dependency:Dependency` to define the libraries on which the experiment needs to run.


# A round is defined by a local training step followed by an aggregation operation
NUM_ROUNDS = 3

# The Dependency object is instantiated in order to install the right libraries in
# the Python environment of each organization.
# The local dependencies are local packages to be installed using the command `pip install -e .`.
# Flamby is a local dependency. We put as argument the path to the `setup.py` file.
algo_deps = Dependency(pypi_dependencies=["torch==1.11.0"], local_dependencies=[
    pathlib.Path.cwd().parent.parent])

compute_plan = execute_experiment(
    client=algo_provider_client,
    algo=MyAlgo(),
    strategy=strategy,
    train_data_nodes=train_data_nodes,
    evaluation_strategy=my_eval_strategy,
    aggregation_node=aggregation_node,
    num_rounds=NUM_ROUNDS,
    experiment_folder=str(pathlib.Path.cwd() / "experiment_summaries"),
    dependencies=algo_deps,
)

# %%
# Explore the results
# *******************

# %%
# List results
# ============

performance_df = pd.DataFrame(client.get_performances(compute_plan.key).dict())
print("\nPerformance Table: \n")
print(performance_df[["worker", "round_idx", "performance"]])


# %%
# Plot results
# ============
plt.title("Test dataset results")
plt.xlabel("Rounds")
plt.ylabel("Metric")


for i, id in enumerate(DATA_PROVIDER_ORGS_ID):
    df = performance_df.query(f"worker == '{id}'")
    plt.plot(df["round_idx"], df["performance"], label=f"Client {i} ({id})")

# plt.legend(loc=(1.1, 0.3), title="Test set")
plt.legend(loc="lower right")
plt.show()


# %%
# Download a model
# ================
#
# After the experiment, you might be interested in downloading your trained model.
# To do so, you will need the source code in order to reload your code architecture in memory.
# You have the option to choose the client and the round you are interested in downloading.
#
# If `round_idx` is set to `None`, the last round will be selected by default.


client_to_dowload_from = DATA_PROVIDER_ORGS_ID[0]
round_idx = None

folder = str(pathlib.Path.cwd() / "experiment_summaries" /
             compute_plan.key / ALGO_ORG_ID / (round_idx or "last"))

download_algo_files(
    client=data_provider_clients[client_to_dowload_from],
    compute_plan_key=compute_plan.key,
    round_idx=round_idx,
    dest_folder=folder,
)

model = load_algo(input_folder=folder)._model

print(model)
print([p for p in model.parameters()])
