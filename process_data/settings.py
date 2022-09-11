class TaskSettings(object):
    def __init__(self):
        self.train_settings = {
            "ag_news": {
                'epoch': 5, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 500, 'max_length': 128,
                'augweight': 0.5
            },
            "imdb": {
                'epoch': 5, 'batch_size': 8, 'aug_batch_size': 8, 'val_steps': 500, 'max_length': 512,
                'augweight': 0.5
            },
            "mnli": {
                'epoch': 5, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 100, 'max_length': 128,
                'augweight': 0.2
            },
            "mrpc": {
                'epoch': 10, 'batch_size': 32, 'aug_batch_size': 32, 'val_steps': 50, 'max_length': 128,
                'augweight': 0.2
            },
            "qnli": {
                'epoch': 5, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 100, 'max_length': 128,
                'augweight': 0.2
            },
            "qqp": {
                'epoch': 5, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 300, 'max_length': 128,
                'augweight': 0.2
            },
            "rte": {
                'epoch': 10, 'batch_size': 32, 'aug_batch_size': 32, 'val_steps': 50, 'max_length': 128,
                'augweight': -0.2
            },
            "sst": {
                'epoch': 5, 'batch_size': 48, 'aug_batch_size': 48, 'val_steps': 200, 'max_length': 128,
                'augweight': 0.5
            },
            "sst2": {
                'epoch': 5, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 100, 'max_length': 128,
                'augweight': 0.5
            },
            "trec": {
                'epoch': 20, 'batch_size': 96, 'aug_batch_size': 96, 'val_steps': 100, 'max_length': 128,
                'augweight': 0.5
            },
        }
        self.task_to_keys = {
            "addprim_jump": ["commands"],
            "addprim_turn_left": ["commands"],
            "ag_news": ["text"],
            "anli": ["premise", "hypothesis"],
            "imdb": ["text"],
            "mnli": ["premise", "hypothesis"],
            "mrpc": ["sentence1", "sentence2"],
            "qnli": ["question", "sentence"],
            "qqp": ["question1", "question2"],
            "rte": ["sentence1", "sentence2"],
            "sst": ["sentence"],
            "sst2": ["sentence"],
            "trec": ["text"],
        }
        self.pair_datasets = ['mnli', 'mrpc', 'qnli', 'qqp', 'rte',]
        self.SCAN = ['addprim_jump', 'addprim_turn_left', 'simple']
        self.low_resource = {
            "ag_news": [0.01, 0.02, 0.05, 0.1, 0.2],
            "sst": [0.01, 0.02, 0.05, 0.1, 0.2],
            "sst2": [0.01, 0.02, 0.05, 0.1, 0.2]
        }
