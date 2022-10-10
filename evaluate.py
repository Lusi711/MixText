import argparse
import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from process_data.Load_data import DataProcess
from run import load_model


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--mode', type=str, choices=['raw', 'aug', 'raw_aug', 'visualize'], required=True)
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--num_proc', type=int, default=8, help='multi process number used in dataloader process')

    # training settings
    parser.add_argument('--seed', default=42, type=int, help='seed ')
    parser.add_argument('--batch_size', default=128, type=int, help='train examples in each batch')
    parser.add_argument('--max_length', default=128, type=int, help='encode max length')
    parser.add_argument('--label_name', type=str, default='label')
    parser.add_argument('--model', type=str, default='roberta-base')
    parser.add_argument('--low_resource_dir', type=str, help='Low resource data dir')

    # train on augmentation dataset parameters
    parser.add_argument('--aug_batch_size', default=128, type=int, help='train examples in each batch')
    parser.add_argument('--augweight', default=0.2, type=float)
    parser.add_argument('--data_path', type=str, help="augmentation file path")
    parser.add_argument(
        '--min_train_token', type=int, default=0, help="minimum token num restriction for train dataset"
    )
    parser.add_argument(
        '--max_train_token', type=int, default=0, help="maximum token num restriction for train dataset"
    )
    parser.add_argument('--mix', action='store_false', help='train on 01mixup')

    # random mixup
    parser.add_argument('--alpha', type=float, default=0.1, help="online augmentation alpha")
    parser.add_argument('--onlyaug', action='store_true', help="train only on online aug batch")
    parser.add_argument('--difflen', action='store_true', help="train only on online aug batch")
    parser.add_argument('--random_mix', type=str, help="random mixup ")

    # visualize dataset

    args = parser.parse_args()
    if args.data == 'trec':
        assert args.label_name in ['label-fine', 'label-coarse'], \
            "If you want to train on trec dataset with augmentation, you have to name the label of split"
        if not args.output_dir:
            args.output_dir = os.path.join('DATA', args.data.upper(), 'runs', args.label_name, args.mode)
    if args.mode == 'raw':
        args.batch_size = 64
    if 'aug' in args.mode:
        assert args.data_path
        if args.mode == 'aug':
            args.seed = 42
    if args.data in ['rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
        args.task = 'pair'
    else:
        args.task = 'single'

    return args


def load_data(args):
    data_process = DataProcess(args)

    print('=' * 20, 'Start processing dataset', '=' * 20)
    t1 = time.time()
    dataloader, nlabels = data_process.test_data(count_label=True)
    t2 = time.time()
    print('=' * 20, 'Dataset process done! cost {:.2f}s'.format(t2 - t1), '=' * 20)

    return dataloader, nlabels


def eval_logits(model, dataloader):
    model.eval()  # evaluation after each epoch
    logits = []
    label_ids = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch in bar:
        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(**batch)
            batch_logits = outputs.logits
            logits.append(batch_logits)
            label_ids.append(batch['labels'].detach().cpu().numpy())

    label_ids = np.hstack(label_ids).flatten()
    logits = torch.cat(logits, dim=0)
    logits = logits.detach().cpu().numpy()

    return label_ids, logits


def test(model, dataloader):
    y_true, logits = eval_logits(model, dataloader)
    y_pred = np.argmax(logits, axis=1).flatten()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    args = parse_argument()
    test_dataloader, label_num = load_data(args)
    model = load_model(args, label_num)
    test(model, test_dataloader)
