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
    def __init__(self, model, use_mapper, mapper_classify):
        super().__init__()
        self.base_model = model
        self.handler = self.base_model.handler
        self.norm = NormLayer()
        self.use_mapper = use_mapper
        self.mapper_classify = mapper_classify
        
    def forward(self, x):
        orig_out = self.base_model(x)
        cont_feats = self.handler.get_all_features('downscale','downscale')
        cont_feats = cont_feats.reshape(*cont_feats.shape[:-3], -1)
        if self.use_mapper:
            feats = self.base_model._mapper(cont_feats)
        else:
            feats = self.norm(cont_feats)
        
        if self.mapper and self.mapper_classify:
            out = orig_out*0 + self.base_model._mapper_classifier(feats.detach())
        else : 
            out = orig_out
        return out, feats
        
class ContrastTrainer(BaseTrainer):
    def __init__(self, train_configs):
        super().__init__(train_configs)
        self.attach_extractor()
        self.model = ContrastiveWrapper(self.model)
        
    def attach_extractor(self):
        target_layer = self.model_configs.contrastive_layer
        # Automatically assume last conv feat if target_layer is None
        self.model.handler = FGHandler(self.model, target_layer)
        if self.configs.train.mapper is not None:
            dummy_loader = DataLoader(self.val_dataset, num_workers=0)
            for batch_info in tqdm(dummy_loader):
                if isinstance(batch_info, list):
                    taglist = ['input', 'label1', 'label2', 'lam', 'phase']
                    batch_info = {k : v for k,v in zip(taglist, batch_info[:len(taglist)])}
                dummy_out = self.model(batch_info['input'])
                dummy_feat = self.model.handler.get_all_features('downscale','downscale')
                shape = dummy_feat.reshape(*dummy_feat.shape[:-3], -1).shape[-1]
                break
            #TODO make mapper controllable with custom model
            self.model._mapper = torch.nn.Sequential(
                torch.nn.LayerNorm(shape),
                torch.nn.Linear(shape, 128),
                NormLayer()
            )
            if self.configs.train.mapper.classify:
                self.model._mapper_classifier = torch.nn.Linear(128, dummy_out.shape[-1])
            
        