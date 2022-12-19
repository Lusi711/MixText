import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk, Features, Value
from matplotlib import pyplot as plt
from transformers import RobertaTokenizer
from process_data import settings


class DataProcess(object):
    def __init__(self, args=None):
        if args:
            print('Initializing with args')
            self.data = args.data if args.data else None
            self.task = args.task if args.task else None
            self.class_type = args.class_type if args.class_type else 'multiclass'
            self.tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=True) if args.model else None
            self.task_settings = settings.TaskSettings()
            self.max_length = args.max_length if args.max_length else None
            self.label_name = args.label_name if args.label_name else None
            self.batch_size = args.batch_size if args.batch_size else None
            self.aug_batch_size = args.aug_batch_size if args.aug_batch_size else None
            self.min_train_token = args.min_train_token if args.min_train_token else None
            self.max_train_token = args.max_train_token if args.max_train_token else None
            self.num_proc = args.num_proc if args.num_proc else None
            self.low_resource_dir = args.low_resource_dir if args.low_resource_dir else None
            self.data_path = args.data_path if args.data_path else None
            self.random_mix = args.random_mix if args.random_mix else None

    def encode(self, examples):
        return self.tokenizer(examples[self.task_settings.task_to_keys[self.data][0]], max_length=self.max_length,
                              truncation=True, padding='max_length')

    def encode_pair(self, examples):
        return self.tokenizer(examples[self.task_settings.task_to_keys[self.data][0]],
                              examples[self.task_settings.task_to_keys[self.data][1]], max_length=self.max_length,
                              truncation=True, padding='max_length')

    def validation_data(self):
        validation_set = self.validation_dataset(self.data, self.class_type)
        print('=' * 20, 'multiprocess processing validation dataset', '=' * 20)
        # Process dataset to make dataloader
        if self.task == 'single':
            validation_set = validation_set.map(self.encode, batched=True, num_proc=self.num_proc)
        else:
            validation_set = validation_set.map(self.encode_pair, batched=True, num_proc=self.num_proc)
        validation_set = validation_set.rename_column(self.label_name, "labels")
        validation_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        return validation_set

    def train_data(self, count_label=False):
        train_set, label_num = self.train_dataset(
            data=self.data, class_type=self.class_type, low_resource_dir=self.low_resource_dir, label_num=count_label,
        )
        print('=' * 20, 'multiprocess processing train dataset', '=' * 20)
        if self.task == 'single':
            train_set = train_set.map(self.encode, batched=True, num_proc=self.num_proc)
        else:
            train_set = train_set.map(self.encode_pair, batched=True, num_proc=self.num_proc)
        if self.random_mix:
            # sort the train dataset
            print('-' * 20, 'random_mixup', '-' * 20)
            train_set = train_set.map(lambda examples: {'token_num': np.sum(np.array(examples['attention_mask']))})
            train_set = train_set.sort('token_num', reverse=True)
        train_set = train_set.rename_column(self.label_name, "labels")
        if self.min_train_token:
            print('-' * 20, 'filter sample whose sentence shorter than {}'.format(self.min_train_token), '-' * 20)
            train_set = train_set.filter(lambda example: sum(example['attention_mask']) > self.min_train_token + 2)
        if self.max_train_token:
            print('-' * 20, 'filter sample whose sentence longer than {}'.format(self.max_train_token), '-' * 20)
            train_set = train_set.filter(lambda example: sum(example['attention_mask']) < self.max_train_token + 2)
        train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        if count_label:
            return train_set, label_num
        else:
            return train_set, 1

    def augmentation_data(self, count_label=False):
        try:
            aug_dataset = load_dataset('csv', data_files=[self.data_path])['train']
        except:
            aug_dataset = load_from_disk(self.data_path)

        if self.class_type == 'multiclass':
            label_num = len(aug_dataset[self.label_name][0])
        else:
            label_num = 1
        print('=' * 20, 'multiprocess processing aug dataset', '=' * 20)
        if self.task == 'single':
            aug_dataset = aug_dataset.map(self.encode, batched=True, num_proc=self.num_proc)
        else:
            aug_dataset = aug_dataset.map(self.encode_pair, batched=True, num_proc=self.num_proc)
        aug_dataset = aug_dataset.rename_column(self.label_name, 'labels')
        aug_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if count_label:
            return aug_dataset, label_num
        else:
            return aug_dataset

    def test_data(self, count_label=False):
        if self.data in ['sst2', 'rte', 'qqp', 'mnli', 'qnli']:
            test_set = self.validation_dataset(data=self.data, class_type=self.class_type)
            label_num = len(set(test_set[self.label_name]))
        elif self.data in ['ag_news', 'imdb', 'mrpc', 'sst', 'trec',]:
            test_set, label_num = self.test_dataset(self.data, self.class_type, label_num=count_label)
        print('=' * 20, 'multiprocess processing test dataset', '=' * 20)

        # Process dataset to make dataloader
        if self.task == 'single':
            test_set = test_set.map(self.encode, batched=True, num_proc=self.num_proc)
        else:
            test_set = test_set.map(self.encode_pair, batched=True, num_proc=self.num_proc)
        test_set = test_set.rename_column(self.label_name, "labels")
        test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

        if count_label:
            return test_dataloader, label_num
        else:
            return test_dataloader

    def validation_dataset(self, data, class_type):
        if data in ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
            if data == 'mnli':
                validation_set = load_dataset('glue', data, split='validation_mismatched')
            else:
                validation_set = load_dataset('glue', data, split='validation')
            print('-' * 20, 'Validation on glue@{}'.format(data), '-' * 20)
        elif data in ['imdb', 'ag_news', 'trec']:
            validation_set = load_dataset(data, split='test')
            print('-' * 20, 'Validation on {}'.format(data), '-' * 20)
        elif data == 'sst':
            validation_set = load_dataset(data, 'default', split='validation')
            validation_set = validation_set.map(
                lambda example: {
                    'label': int(example['label'] * 10 // 2) if example['label'] != 1 else 4
                }, remove_columns=['tokens', 'tree'], num_proc=self.num_proc
            )
            if class_type == 'multiclass':
                validation_set = validation_set.cast(
                    Features({'sentence': Value(dtype='string', id=None), "label": Value("int32")}),
                    batch_size=None, load_from_cache_file=False,
                )
            print('-' * 20, 'Validation on {}'.format(data), '-' * 20)
        else:
            validation_set = load_dataset(data, split='validation')
            print('-' * 20, 'Validation on {}'.format(data), '-' * 20)

        return validation_set

    def train_dataset(self, data, class_type, low_resource_dir=None, split='train', label_num=False):
        if low_resource_dir:
            train_set = load_from_disk(os.path.join(low_resource_dir, 'partial_train'))
        else:
            if data in ['sst2', 'rte', 'mrpc', 'qqp', 'mnli', 'qnli']:
                train_set = load_dataset('glue', data, split=split)
            elif data == 'sst':
                train_set = load_dataset(data, 'default', split=split)
                train_set = train_set.map(
                    lambda example: {
                        'label': int(example['label'] * 10 // 2) if example['label'] != 1 else 4
                    }, remove_columns=['tokens', 'tree'], num_proc=self.num_proc
                )
                if class_type == 'multiclass':
                    train_set = train_set.cast(
                        Features({'sentence': Value(dtype='string', id=None), "label": Value("int32")}),
                        batch_size=None, load_from_cache_file=False,
                    )
            else:
                train_set = load_dataset(data, split=split)

        if label_num:
            try:
                return train_set, len(set(train_set[self.label_name]))
            except KeyError:
                return train_set, len(train_set[self.label_name][0])
        else:
            return train_set, 1

    def test_dataset(self, data, class_type, label_num):
        if data == 'mrpc':
            test_set = load_dataset('glue', data, split='test')
            print('-' * 20, 'Test on glue@{}'.format(data), '-' * 20)
        elif data in ['imdb', 'ag_news', 'trec']:
            test_set = load_dataset(data, split='test')
            print('-' * 20, 'Test on {}'.format(data), '-' * 20)
        elif data == 'sst':
            test_set = load_dataset(data, 'default', split='test')
            test_set = test_set.map(
                lambda example: {
                    'label': int(example['label'] * 10 // 2) if example['label'] != 1 else 4
                }, remove_columns=['tokens', 'tree'], num_proc=self.num_proc
            )
            if class_type == 'multiclass':
                test_set = test_set.cast(
                    Features({'sentence': Value(dtype='string', id=None), "label": Value("int32")}), batch_size=None,
                    load_from_cache_file=False,
                )
            print('-' * 20, 'Test on {}'.format(data), '-' * 20)
        else:
            test_set = load_dataset(data, split='test')
            print('-' * 20, 'Test on {}'.format(data), '-' * 20)

        if label_num:
            if class_type == 'multiclass':
                try:
                    return test_set, len(set(test_set[self.label_name]))
                except:
                    return test_set, len(test_set[self.label_name][0])
            elif class_type == 'ordinal':
                return test_set, 1
        else:
            return test_set


if __name__ == "__main__":
    dp = DataProcess()
    dp.label_name = 'label'
    dp.class_type = 'ordinal'
    dp.data_path = 'C:/WuQi/NUS_CEG/Y4S1/CG4001/TreeMix/DATA/SST/generated/times5_min0_seed0_0.3_0.1_20k'
    train_dataset, _ = dp.train_dataset('sst', 'multiclass', label_num=True)
    print("Train dataloader:\n", train_dataset)
    val_dataset = dp.validation_dataset('sst', 'multiclass')
    print("Validation dataloader:\n", val_dataset)
    test_dataset, _ = dp.test_dataset('sst', 'multiclass', label_num=True)
    print("Test dataloader:\n", test_dataset)
    aug_dataset = load_from_disk(dp.data_path)
    aug_dataset = aug_dataset.map(
        lambda example: {
            'label': int(example['label'] * 10 // 2) if example['label'] != 1 else int(example['label'])
        }, remove_columns=['attention_mask', 'input_ids'], num_proc=4
    )
    aug_dataset = aug_dataset.cast(
        Features({"label": Value("int32"), 'sentence': Value(dtype='string', id=None), }), batch_size=None,
        load_from_cache_file=False,
    )
    print("Augmentation dataloader:\n", aug_dataset)

    train_labels, aug_labels = train_dataset['label'], aug_dataset['label']
    classes = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    train_values, aug_values = np.bincount(train_labels), np.bincount(aug_labels)

    N = max(len(train_values), len(aug_values))
    ind = np.arange(N)
    fig = plt.figure(figsize=(10, 5))
    # creating the bar plot
    p1 = plt.bar(ind, train_values, width=0.35)
    p2 = plt.bar(ind, aug_values, width=0.35, bottom=train_values)

    plt.xlabel("Sentiment classes")
    plt.xticks(ind, classes)
    plt.ylabel("Number of samples")
    plt.ylim(0, 9000)
    plt.title("Distribution of training samples across ordinal classes (after sample selection)")
    plt.legend((p1[0], p2[0]), ('Original', 'Augmentation'))
    plt.show()
