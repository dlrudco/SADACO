import torch
from .torchvggish.vggish import VGGish as _VGGish

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

def vggish(**kwargs):
    model = _VGGish(urls=model_urls, **kwargs)
    return model

class VGGish(torch.nn.Module):
    def __init__(self, num_classes, freeze=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.freeze = freeze
        self.base = vggish(**kwargs)
        
        if self.freeze:
            for params in self.base.parameters():
                params.requires_grad = False
    
        self.hidden = (self.num_classes + 128) // 3
        self.classifier = torch.nn.Sequential(
                                        torch.nn.Linear(128, self.hidden * 2),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden*2, self.hidden),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(self.hidden, self.num_classes))
    def forward(self, x):
        # Input format of SADACO follows [N, C=1, Mel, Time]
        # Vggish expects [N, C=1, Time, Mel]
        x = x.permute(0,1,3,2)
        x = self.base.features(x)
        # Tensorflow Compatable format [N, Time, Mel, C]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], -1)
        x = self.base.embeddings(x)
        x = self.classifier(x)
        return x

