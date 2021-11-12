import sys
sys.path.append(r'/home/rudy/Documenti/avalanche')


import torch
from torch.optim.sgd import SGD
from torch.nn.modules.loss import CrossEntropyLoss

from avalanche.models import SimpleMLP
from avalanche.benchmarks.classic import RotatedMNIST
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.evaluation.metrics import *
from avalanche.training.strategies.strategy_wrappers import Naive

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
model = SimpleMLP(num_classes=10)

# Here we show all the MNIST variation we offer in the "classic" benchmarks
# benchmark = PermutedMNIST(n_experiences=5, seed=1)
benchmark = RotatedMNIST(n_experiences=5, rotations_list=[30, 60, 90, 120, 150], seed=1)
# benchmark = SplitMNIST(n_experiences=5, seed=1)

# choose some metrics and evaluation method
interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    ExperienceForgetting(),
    loggers=[interactive_logger])

# Than we can extract the parallel train and test streams
train_stream = benchmark.train_stream
test_stream = benchmark.test_stream

# Prepare for training & testing
optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

# Continual learning strategy
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    device=device, evaluator=eval_plugin
)
#Naive no test_mb_size!
"""
cl_strategy = Naive(
    model, optimizer, criterion, train_mb_size=32, train_epochs=2,
    test_mb_size=32, device=device, evaluator=eval_plugin
)
"""

# train and test loop
results = []
for train_task in train_stream:
    print("Current Classes: ", train_task.classes_in_this_experience)
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.eval(test_stream))