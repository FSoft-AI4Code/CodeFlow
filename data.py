from torchtext import data
from torchtext.data import Iterator
import pandas as pd
import config
import torch

def read_data(data_path, fields):
    csv_data = pd.read_csv(data_path, chunksize=100)
    all_examples = []
    for n, chunk in enumerate(csv_data):
        
        examples = chunk.apply(lambda r: data.Example.fromlist([eval(r['nodes']), eval(r['forward']), eval(r['backward']),
                                                                eval(r['target'])], fields), axis=1)
        all_examples.extend(list(examples))
    return all_examples

def get_iterators(args, device):
    TEXT = data.Field(tokenize=lambda x: x.split()[:args.max_token])
    NODE = data.NestedField(TEXT, preprocessing=lambda x: x[:args.max_node], include_lengths=True)
    ROW = data.Field(pad_token=1.0, use_vocab=False,
                     preprocessing=lambda x: [1, 1] if any(i > args.max_node for i in x) else x)
    EDGE = data.NestedField(ROW)
    TARGET = data.Field(use_vocab=False, preprocessing=lambda x: x[:args.max_node], pad_token=0, batch_first=True)
    
    fields = [("nodes", NODE), ("forward", EDGE), ("backward", EDGE), ("target", TARGET)]

    print('Read data...')
    examples = read_data(f'data/{args.data}_train.csv', fields)
    train = data.Dataset(examples, fields)
    NODE.build_vocab(train, max_size=args.vocab_size)

    examples = read_data(f'data/{args.data}_test.csv', fields)
    test = data.Dataset(examples, fields)
    train_iter = Iterator(train,
                          batch_size=args.batch_size,
                          device=device,
                          sort=False,
                          sort_key=lambda x: len(x.nodes),
                          sort_within_batch=False,
                          repeat=False)
    test_iter = Iterator(test, batch_size=args.batch_size, device=device, train=False,
                         sort=False, sort_key=lambda x: len(x.nodes), sort_within_batch=False, repeat=False, shuffle=False)
    print("Done")
    return train_iter, test_iter
