import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

class DummDataSet(Dataset):
    def __init__(self):
        super().__init__()
        self.data = [np.random.randn( *(np.random.randint(low=2,high=3, size=(3)).tolist())) for _ in range(10)]
        self.labels = [np.random.randint(low=1,high=10) for _ in range(10)]
        
    def __getitem__(self, index):
        return {'data':self.data[index], 'label':self.labels[index]}
    
    def __len__(self):
        return len(self.data)
    
class DummCollator:
    def __init__(self):
        pass
    
    def __call__(self, data):
        return data
        
if __name__ == "__main__":
    my_dataset = DummDataSet()
    my_collate = DummCollator()
    my_loader = DataLoader(dataset=my_dataset, collate_fn=default_collate, num_workers=0, batch_size=2)
    for batch in my_loader:
        breakpoint()
    