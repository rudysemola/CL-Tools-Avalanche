# TODO: run the script...

import sys
sys.path.append(r'/home/rudy/Documenti/avalanche')

"Input script"
import argparse
parser = argparse.ArgumentParser(description='Input CL Ex')
parser.add_argument('hs', metavar='N', type=int,
                    help='# of hidden size of the MLP model')
parser.add_argument('lr', metavar='N', type=float,
                    help='learning rate opt')
parser.add_argument('cuda', metavar='N', type=int,
                    help='cuda or cpu setting')
parser.add_argument('lwf_alpha', metavar='N', type=float,
                    help='hyper: lwf_alpha')
parser.add_argument('softmax_temperature', metavar='N', type=int,
                    help='hyper:lwf - softmax_temperature')
parser.add_argument('epochs', metavar='N', type=int,
                    help='hyper: lwf - epochs')
parser.add_argument('minibatch_size', metavar='N', type=int,
                    help='hyper: lwf - minibatch_size')
args = parser.parse_args()


import torch
from avalanche.models.simple_mlp import SimpleMLP
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.logging.interactive_logging import InteractiveLogger
from avalanche.training.plugins.evaluation import EvaluationPlugin
from avalanche.evaluation.metrics import *
from avalanche.training.strategies.strategy_wrappers import LwF

model = SimpleMLP(hidden_size=args.hs)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# check if selected GPU is available or use CPU
assert args.cuda == -1 or args.cuda >= 0, "cuda must be -1 or >= 0."
if args.cuda >= 0:
    assert torch.cuda.device_count() > args.cuda, f"{args.cuda + 1} GPU needed. Found {torch.cuda.device_count()}."
device = 'cpu' if args.cuda == -1 else f'cuda:{args.cuda}'
print(f'Using device: {device}')

# create split scenario
scenario = SplitMNIST(n_experiences=5, return_task_id=False)

interactive_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    ExperienceForgetting(),
    loggers=[interactive_logger])

# create strategy
assert len(args.lwf_alpha) == 1 or len(args.lwf_alpha) == 5, 'Alpha must be a non-empty list.'
lwf_alpha = args.lwf_alpha[0] if len(args.lwf_alpha) == 1 else args.lwf_alpha

strategy = LwF(model, optimizer, criterion, alpha=lwf_alpha,
               temperature=args.softmax_temperature,
               train_epochs=args.epochs, device=device,
               train_mb_size=args.minibatch_size, evaluator=eval_plugin)

# train on the selected scenario with the chosen strategy
print('Starting experiment...')
results = []
for train_batch_info in scenario.train_stream:
    print("Start training on experience ", train_batch_info.current_experience)

    strategy.train(train_batch_info, num_workers=4)
    print("End training on experience ", train_batch_info.current_experience)
    print('Computing accuracy on the test set')
    results.append(strategy.eval(scenario.test_stream[:]))