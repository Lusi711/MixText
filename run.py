import argparse
import logging
import os
import random
import re
import time
import torch
import torch.nn.parallel
import torch.utils.data.distributed
import online_augmentation

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from itertools import cycle
from multiprocessing import cpu_count
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from process_data.Load_data import DataProcess


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cross_entropy(logits, target):
    p = F.softmax(logits, dim=1)
    log_p = -torch.log(p)
    loss = target * log_p
    batch_num = logits.shape[0]

    return loss.sum() / batch_num


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    return accuracy_score(labels_flat, pred_flat)


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size

    return rt


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--mode', type=str, choices=['raw', 'aug', 'raw_aug', 'visualize'], required=True)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--class_type', type=str, choices=['ordinal', 'multiclass'], help='classification problem')
    parser.add_argument('--num_proc', type=int, help='multi process number used in dataloader process')

    # training settings
    parser.add_argument('--output_dir', type=str, help="tensorboard file output directory")
    parser.add_argument('--epoch', type=int, default=5, help='train epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--seed', default=42, type=int, help='seed ')
    parser.add_argument('--batch_size', default=128, type=int, help='train examples in each batch')
    parser.add_argument('--val_steps', default=100, type=int, help='evaluate on dev datasets every steps')
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
    if not args.num_proc:
        args.num_proc = cpu_count()
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
    if not args.output_dir:
        args.output_dir = os.path.join('DATA', args.data.upper(), 'runs', args.mode)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.data in ['rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
        args.task = 'pair'
    else:
        args.task = 'single'

    return args


def tensorboard_settings(args):
    if 'raw' in args.mode:
        if args.data_path:
            # raw_aug
            log_dir = os.path.join(
                args.output_dir, 'Raw_Aug_{}_{}_{}_{}_{}'.format(
                    args.data_path.split('/')[-1], args.seed, args.augweight, args.batch_size, args.aug_batch_size
                )
            )
            writer = SummaryWriter(log_dir=log_dir)
        else:
            # raw
            if args.random_mix:
                log_dir = os.path.join(
                    args.output_dir, 'Raw_random_mixup_{}_{}_{}'.format(args.random_mix, args.alpha, args.seed)
                )
                writer = SummaryWriter(log_dir=log_dir)
            else:
                log_dir = os.path.join(args.output_dir, 'Raw_{}'.format(args.seed))
                writer = SummaryWriter(log_dir=log_dir)
    elif args.mode == 'aug':
        # aug
        log_dir = os.path.join(
            args.output_dir, 'Aug_{}_{}_{}_{}_{}'.format(
                args.data_path.split('/')[-1], args.seed, args.augweight, args.batch_size, args.aug_batch_size
            )
        )
        writer = SummaryWriter(log_dir=log_dir)

    return writer


def logging_settings(args):
    logger = logging.getLogger('result')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    if not os.path.exists(os.path.join('DATA', args.data.upper(), 'logs')):
        os.makedirs(os.path.join('DATA', args.data.upper(), 'logs'))
    if args.low_resource_dir:
        log_path = os.path.join('DATA', args.data.upper(), 'logs', 'lowresource_best_result.log')
    else:
        log_path = os.path.join('DATA', args.data.upper(), 'logs', 'best_result.log')

    fh = logging.FileHandler(log_path, mode='a+', encoding='utf-8')
    ft = logging.Filter(name='result.a')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    fh.addFilter(ft)
    logger.addHandler(fh)
    result_logger = logging.getLogger('result.a')

    return result_logger


def load_model(args, label_num):
    t1 = time.time()
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will t ake care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1  # the number of gpu on each proc
    args.device = device
    if args.local_rank != -1:
        args.world_size = torch.cuda.device_count()
    else:
        args.world_size = 1
    print('*' * 40, '\nSettings:{}'.format(args))
    print('*' * 40)
    print('=' * 20, 'Loading models', '=' * 20)
    model = RobertaForSequenceClassification.from_pretrained(args.model, num_labels=label_num)
    model.to(device)
    t2 = time.time()
    print('=' * 20, 'Loading models complete!, cost {:.2f}s'.format(t2 - t1), '=' * 20)
    # model parallel
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    elif args.n_gpu > 1:
        model = nn.DataParallel(model)
    if args.load_model_path is not None:
        print("=" * 20, "Load model from ", args.load_model_path)
        model.load_state_dict(torch.load(args.load_model_path, map_location=args.device))

    return model


def filter_aug_dataset(args, model, aug_dataset):
    def sharpen_augment_labels(examples):
        mix_label = examples['labels'].detach().cpu().numpy()
        batch = examples.copy()
        if args.mix:
            with torch.no_grad():
                del batch['labels']
                outputs = model(**batch)
                logits = outputs.logits
        else:
            with torch.no_grad():
                outputs = model(**batch)
                logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        if args.class_type == 'multiclass':
            y_pred = np.argmax(logits, axis=0)
            selected_indices = np.where(y_pred.flatten() != mix_label)[0]
        elif args.class_type == 'ordinal':
            selected_indices = np.where(np.abs(logits.flatten() - mix_label) > 0.10)
        examples['labels'][selected_indices] = float('nan')
        for key in examples:
            examples[key] = examples[key].numpy()
        return examples

    if args.data_path:
        aug_dataset = aug_dataset.map(
            sharpen_augment_labels, batched=True, batch_size=args.batch_size, load_from_cache_file=False,
        )
        indices_to_keep = set(np.arange(len(aug_dataset['labels']))) - set(np.argwhere(np.isnan(
            aug_dataset['labels'].numpy()))[:, 0])
        aug_dataset = aug_dataset.select(indices_to_keep)
        aug_dataset = aug_dataset.rename_column('labels', args.label_name)
        save_path = os.path.join(
            'DATA', 'SST', 'generated', 'times5_min0_seed0_0.3_0.1_{}k'.format(round(len(aug_dataset) // 1000, -1))
        )
        aug_dataset.save_to_disk(save_path)

    return aug_dataset


def train(args):
    # ========================================
    #         Tensorboard &Logging
    # ========================================
    writer = tensorboard_settings(args)
    result_logger = logging_settings(args)
    data_process = DataProcess(args)
    # ========================================
    #             Loading datasets
    # ========================================
    print('=' * 20, 'Start processing dataset', '=' * 20)
    t1 = time.time()

    validation_set = data_process.validation_data()
    val_dataloader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)

    if args.mode != 'aug':
        train_set, label_num = data_process.train_data(count_label=False)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    if args.data_path:
        print('=' * 20, 'Train Augmentation dataset path: {}'.format(args.data_path), '=' * 20)
        aug_dataset, label_num = data_process.augmentation_data(count_label=True)
        aug_dataloader = torch.utils.data.DataLoader(aug_dataset, batch_size=args.aug_batch_size, shuffle=True)

    t2 = time.time()
    print('=' * 20, 'Dataset process done! cost {:.2f}s'.format(t2 - t1), '=' * 20)

    # ========================================
    #                   Model
    # ========================================
    model = load_model(args, label_num)

    # ========================================
    #     Assess Confidence of Augmentation
    # ========================================
    if args.data_path:
        aug_dataset = filter_aug_dataset(args, model, aug_dataset)
        aug_dataloader = torch.utils.data.DataLoader(aug_dataset, batch_size=args.aug_batch_size, shuffle=True)
        if args.mode == 'aug':
            train_dataloader = aug_dataloader
        else:
            aug_dataloader = cycle(aug_dataloader)
    # ========================================
    #           Optimizer Settings
    # ========================================
    optimizer = AdamW(model.parameters(), lr=args.lr)
    all_steps = args.epoch * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=all_steps)
    if args.class_type == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    elif args.class_type == 'ordinal':
        criterion = nn.MSELoss()

    # ========================================
    #               Train
    # ========================================
    model.train()
    print('=' * 20, 'Start training', '=' * 20)
    best_loss, best_acc = validate(args, model, val_dataloader)
    best_steps = 0
    args.val_steps = min(len(train_dataloader), args.val_steps)
    for epoch in range(args.epoch):
        bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader) // args.world_size)
        fail = 0
        for step, batch in bar:
            model.zero_grad()
            # ----------------------------------------------
            #               Train_dataloader
            # ----------------------------------------------
            if args.random_mix:
                try:
                    input_ids, target_a = batch['input_ids'], batch['labels']
                    lam = np.random.choice([0, 0.1, 0.2, 0.3])
                    exchanged_ids, new_index = online_augmentation.random_mixup(args, input_ids, target_a, lam)
                    target_b = target_a[new_index]
                    outputs = model(exchanged_ids.to(args.device), token_type_ids=None,
                                    attention_mask=(exchanged_ids > 0).to(args.device))
                    logits = outputs.logits
                    loss = criterion(logits.to(args.device), target_a.to(args.device)) * (1 - lam) + criterion(
                        logits.to(args.device), target_b.to(args.device)) * lam
                except:
                    fail += 1
                    batch = {k: v.to(args.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
            elif args.mode == 'aug':
                # train only on augmentation dataset
                batch = {k: v.to(args.device) for k, v in batch.items()}
                if args.mix:
                    # train on 01 tree mixup augmentation dataset
                    mix_label = batch['labels']
                    del batch['labels']
                    outputs = model(**batch)
                    logits = outputs.logits
                    loss = cross_entropy(logits, mix_label)
                else:
                    # train on 00&11 tree mixup augmentation dataset
                    outputs = model(**batch)
                    loss = outputs.loss
            else:
                # normal train
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
            # ----------------------------------------------
            #               Aug_dataloader
            # ----------------------------------------------
            if args.mode == 'raw_aug':
                aug_batch = next(aug_dataloader)
                aug_batch = {k: v.to(args.device) for k, v in aug_batch.items()}
                if args.mix:
                    mix_label = aug_batch['labels']
                    del aug_batch['labels']
                    aug_outputs = model(**aug_batch)
                    aug_logits = aug_outputs.logits
                    aug_loss = cross_entropy(aug_logits, mix_label)
                else:
                    aug_outputs = model(**aug_batch)
                    aug_loss = aug_outputs.loss
                loss += aug_loss * args.augweight  # for sst2,rte reaches best performance

            # Backward propagation
            if args.n_gpu > 1:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if args.local_rank == 0 or args.local_rank == -1:
                writer.add_scalar("Loss/loss", loss, step + epoch * len(train_dataloader))
                writer.flush()
                if args.random_mix:
                    bar.set_description(
                        '| Epoch: {:<2}/{:<2}| Best acc:{:.2f}| Fail:{}|'.format(
                            epoch, args.epoch, best_acc * 100, fail
                        )
                    )
                else:
                    bar.set_description(
                        '| Epoch: {:<2}/{:<2}| Best acc:{:.2f}| Best loss: {:.4f}'.format(
                            epoch, args.epoch, best_acc * 100, best_loss
                        )
                    )

            # =================================================
            #                     Validation
            # =================================================
            if (epoch * len(train_dataloader) + step + 1) % args.val_steps == 0:
                avg_val_loss, avg_val_accuracy = validate(args, model, val_dataloader)
                if args.class_type == 'multiclass':
                    validation_criterion, reference_score = avg_val_accuracy, best_acc
                elif args.class_type == 'ordinal':
                    validation_criterion, reference_score = -avg_val_loss, -best_loss

                if validation_criterion > reference_score:
                    best_loss, best_acc = avg_val_loss, avg_val_accuracy
                    best_steps = (epoch * len(train_dataloader) + step) * args.batch_size
                    if args.save_model:
                        torch.save(model.state_dict(), 'best_model.pt')

                if args.local_rank == 0 or args.local_rank == -1:
                    writer.add_scalar("Test/Loss", avg_val_loss, epoch * len(train_dataloader) + step)
                    writer.add_scalar("Test/Accuracy", avg_val_accuracy, epoch * len(train_dataloader) + step)
                    writer.flush()

    if args.data_path:
        aug_num = args.data_path.split('_')[-1]
        if args.low_resource_dir:
            # low resource raw_aug
            partial = re.findall(r'low_resource_(0.\d+)', args.low_resource_dir)[0]
            aug_num_seed = aug_num + '_' + str(args.seed)
            result_logger.info('-' * 160)
            result_logger.info(
                '| Data : {} | Mode: {:<8} | #Aug {:<6} | Best acc:{} | Steps:{} | Weight {} |Aug data: {}'.format(
                    args.data + '_' + partial, args.mode, aug_num_seed, round(best_acc * 100, 3), best_steps,
                    args.augweight, args.data_path
                )
            )
        else:
            # raw_aug
            aug_data_seed = re.findall(r'seed(\d)', args.data_path)[0]
            aug_num_seed = aug_num + '_' + aug_data_seed
            result_logger.info('-' * 160)
            result_logger.info(
                '| Data : {} | Mode: {:<8} | #Aug {:<6} | Best acc:{} | Best loss: {} | Steps:{} | Weight {} | '
                'Aug data: {}'.format(
                    args.data, args.mode, aug_num_seed, round(best_acc * 100, 3), best_loss, best_steps, args.augweight,
                    args.data_path
                )
            )
    else:
        if args.low_resource_dir:
            # low resource raw
            partial = re.findall(r'low_resource_(0.\d+)', args.low_resource_dir)[0]
            result_logger.info('-' * 160)
            result_logger.info(
                '| Data : {} | Mode: {:.8} | Seed: {} | Best acc:{} | Steps:{} | Random mix: {} | Aug data: {}'.format(
                    args.data + '-' + partial, args.mode, args.seed, round(best_acc * 100, 3), best_steps,
                    bool(args.random_mix), args.data_path
                )
            )
        else:
            # raw
            result_logger.info('-' * 160)
            result_logger.info(
                '| Data : {} | Mode: {:.8} | Seed: {} | Best acc:{} | Steps:{} | Random mix: {} | Aug data: {}'.format(
                    args.data, args.mode, args.seed, round(best_acc * 100, 3), best_steps, bool(args.random_mix),
                    args.data_path
                )
            )


def eval_logits(args, model, dataloader, mode='val'):
    total_val_loss = 0
    model.eval()
    logits = []
    label_ids = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if mode == 'aug':
                if args.mix:
                    mix_label = batch['labels']
                    label_ids.append(mix_label)
                    del batch['labels']
                    outputs = model(**batch)
                    batch_logits = outputs.logits
                    loss = cross_entropy(batch_logits, mix_label)
                else:
                    # same label tree mixup augmentation dataset
                    outputs = model(**batch)
                    batch_logits = outputs.logits
                    loss = outputs.loss
            else:
                label_ids.append(batch['labels'].detach().cpu().numpy())
                outputs = model(**batch)
                batch_logits = outputs.logits
                loss = outputs.loss

            if args.n_gpu > 1:
                loss = loss.mean()
            logits.append(batch_logits)

            if args.local_rank != -1:
                torch.distributed.barrier()
                reduced_loss = reduce_tensor(loss, args)
                total_val_loss += reduced_loss
            else:
                total_val_loss += loss.item()

    if label_ids[0].ndim > 1:
        label_ids = np.vstack(label_ids)
    else:
        label_ids = np.hstack(label_ids)
    logits = torch.cat(logits, dim=0)
    logits = logits.detach().cpu().numpy()

    return label_ids, logits, total_val_loss


def validate(args, model, val_dataloader):
    y_true, logits, total_val_loss = eval_logits(args, model, val_dataloader)
    if args.class_type == 'multiclass':
        avg_val_accuracy = flat_accuracy(logits, y_true)
        avg_val_loss = total_val_loss * args.batch_size / len(logits)
    elif args.class_type == 'ordinal':
        y_pred = [np.floor(score) if (score * 10) % 10 < 5 else np.ceil(score) for score in logits.flatten()]
        avg_val_accuracy = sum(y_pred == y_true) / len(y_true)
        avg_val_loss = mean_squared_error(y_true, logits.flatten())

    if args.local_rank != -1:
        avg_val_accuracy = torch.tensor(avg_val_accuracy).to(args.device)
        avg_val_accuracy = reduce_tensor(avg_val_accuracy, args)

    return avg_val_loss, avg_val_accuracy


def test(args):
    data_process = DataProcess(args)
    print('=' * 20, 'Start processing test dataset', '=' * 20)
    t1 = time.time()
    test_dataloader, label_num = data_process.test_data(count_label=True)
    t2 = time.time()
    print('=' * 20, 'Dataset process done! cost {:.2f}s'.format(t2 - t1), '=' * 20)
    print(' - ' * 160)
    args.load_model_path = 'best_model.pt'
    model = load_model(args, label_num)
    avg_test_loss, avg_test_accuracy = validate(args, model, test_dataloader)
    print(
        '| Data : {} | Mode: {:.8} | Seed: {} | Best acc:{} | Best loss: {}'.format(
            args.data, args.mode, args.seed, round(avg_test_accuracy * 100, 3), avg_test_loss,
            args.data_path
        )
    )


def run(args):
    set_seed(args.seed)
    if args.mode in ['raw', 'raw_aug', 'aug']:
        if args.low_resource_dir:
            print("=" * 20, ' Low-resource ', '=' * 20)
        train(args)
        test(args)


if __name__ == '__main__':
    args = parse_argument()
    run(args)
