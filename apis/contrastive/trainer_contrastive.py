from apis.traintest import BaseTrainer
from apis.explain.hookman import FGHandler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class NormLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer=torch.nn.Identity()
    def forward(self,x):
        return torch.nn.functional.normalize(self.layer(x), dim=-1, p=2)
    
class ContrastiveWrapper(torch.nn.Module):
    def __init__(self, model, use_mapper, mapper_classify, keepdims):
        super().__init__()
        self.base_model = model
        self.handler = self.base_model.handler
        self.norm = NormLayer()
        self.use_mapper = use_mapper
        self.mapper_classify = mapper_classify
        self.keepdims = keepdims
    
    def to(self, device):
        self.device = device
        self.base_model.to(device)
        if self.use_mapper:
            self.base_model._mapper.to(device)
            if self.mapper_classify:
                self.base_model._mapper_classifier.to(device)
        else:
            self.norm.to(device)
        
    def forward(self, x):
        orig_out = self.base_model(x)
        cont_feats = self.handler.get_all_features()[0]
        cont_feats = cont_feats.reshape(*cont_feats.shape[:self.keepdims], -1)
        if self.use_mapper:
            feats = self.base_model._mapper(cont_feats.to(self.device))
        else:
            feats = self.norm(cont_feats.to(self.device))
        
        if self.use_mapper and self.mapper_classify:
            out = orig_out*0 + self.base_model._mapper_classifier(feats.detach())
        else : 
            out = orig_out
        
        if self.training:
            return out, feats
        else:
            return out
        
class ContrastTrainer(BaseTrainer):
    def __init__(self, train_configs):
        super().__init__(train_configs)
        
    def attach_extractor(self):
        target_layer = self.model_configs.contrastive_layer
        # Automatically assume last conv feat if target_layer is None
        self.model.handler = FGHandler(self.model, target_layer)
        if self.configs.train.mapper is not None:
            self.keepdims = self.configs.train.mapper.keepdims
            dummy_loader = DataLoader(self.val_dataset, num_workers=0)
            for batch_info in dummy_loader:
                if isinstance(batch_info, list):
                    taglist = ['input', 'label1', 'label2', 'lam', 'phase']
                    batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
                if self.preproc is not None:
                    self.preproc.to(batch_info['input'].device)
                    inputs = self.preproc(batch_info)
                else:
                    inputs = batch_info
                self.model.to(batch_info['input'].device)
                dummy_out = self.model(inputs['input'])
                dummy_feat = self.model.handler.get_all_features()[0]
                shape = dummy_feat.reshape(*dummy_feat.shape[:self.keepdims], -1).shape[-1]
                self.model.handler.reset_all()
                break
            #TODO make mapper controllable with custom model
            self.model._mapper = torch.nn.Sequential(
                torch.nn.Linear(shape, (shape+128)//2),
                torch.nn.ReLU(),
                NormLayer(),
                torch.nn.Linear((shape+128)//2, 128),
                NormLayer()
            )
            if self.configs.train.mapper.classify:
                self.model._mapper_classifier = torch.nn.Linear(128, dummy_out.shape[-1])
            
    def wrap_model(self):
        self.model = ContrastiveWrapper(self.model, use_mapper=self.configs.train.mapper.use,
                                        mapper_classify=self.configs.train.mapper.classify,
                                        keepdims = self.keepdims)