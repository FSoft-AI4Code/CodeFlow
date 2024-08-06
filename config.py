import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="a dummy argument to fool ipython", default="1"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Input batch size for training"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Dimension of hidden states"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=100000, help="Vocab size for training"
    )
    parser.add_argument(
        "--max_node", type=int, default=100, help="Maximum number of nodes"
    )
    parser.add_argument(
        "--max_token", type=int, default=512, help="Maximum number of tokens"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--num_epoch", type=int, default=600, help="Epochs for training")
    parser.add_argument(
        "--cpu", action="store_true", default=False, help="Disables CUDA training"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for CUDA training")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument("--extra_aggregate", action="store_true", default=False)
    parser.add_argument("--delete_redundant_node", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--dryrun", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    # add config epoch that is a list of epochs
    parser.add_argument(
        "--vis_epoch",
        nargs="+",
        type=int,
        default=[],
        help="Epochs for visualization",
    )
    #list of file code to visualize
    parser.add_argument(
        "--vis_code",
        nargs="+",
        type=int,
        default=[],
        help="Code to visualize",
    )
    parser.add_argument("--groundtruth", action="store_true", default=False)
    parser.add_argument("--name_exp", type=str, default=None)
    parser.add_argument("--cuda_num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=300103)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--data", type=str, default="CodeNet")
    parser.add_argument("--runtime_detection", action="store_true", default=False)
    parser.add_argument("--bug_localization", action="store_true", default=False)
    args = parser.parse_args()
    return args