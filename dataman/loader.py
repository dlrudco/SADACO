import torch
from dataman import sampler
from torch.utils.data import DataLoader

def _build_dataloader(dataset, train_configs, data_configs, split):
    if 'sampler' in train_configs.data.dataloader.params.__dict__.keys() and split =='train':
        sampler = getattr(sampler, train_configs.data.sampler.name)(dataset, **train_configs.data.sampler.params)
        assert sampler.batch_size == train_configs.data.dataloader.batch_size, 'sampler parameters should be set to match dataloader.batch_size!'
    else:
        sampler = None
    defaults = {'num_workers' : 8, 'shuffle' : split=='train' and sampler is None, 'drop_last' : split=='train', 'pin_memory' : True, 'persistent_workers' : False}
    defaults.update(train_configs.__dict__[f'{split}_dataloader']['params'])
    dataloader = DataLoader(dataset, batch_sampler=sampler, **defaults)
    return dataloader