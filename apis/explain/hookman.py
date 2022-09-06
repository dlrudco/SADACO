from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            layer_name = name
    return [layer_name]

class FGHandler(object):
    def __init__(self, net, layer_name=None):
        self.net = net
        if layer_name is None:
            self.layer_name = get_last_conv_name(net)
        else:
            if isinstance(layer_name, list):
                self.layer_name = layer_name
            else:
                self.layer_name = [layer_name]
        self.layer_name = self.layer_name + ['module.'+l for l in self.layer_name]
        self.feature = {}
        self.gradient = {}
        for layer in self.layer_name:
            self.feature[layer] = {}
            self.gradient[layer] = {}
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, name):
        def hook(module, input, output):
            self.feature[f'{name}'][f'{input[0].device}'] = output.cpu()

        return hook
        

    def _get_grads_hook(self, name):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple, length == 1
        :return:
        """
        def hook(module, input_grad, output_grad):
            self.gradient[f'{name}'][f'{input_grad[0].device}'] = output_grad[0].cpu()

        return hook
            

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if isinstance(self.layer_name, list):
                if name in self.layer_name:
                    self.handlers.append(module.register_forward_hook(self._get_features_hook(name)))
                    self.handlers.append(module.register_full_backward_hook(self._get_grads_hook(name)))
            else:
                if name == self.layer_name:
                    self.handlers.append(module.register_forward_hook(self._get_features_hook(name)))
                    self.handlers.append(module.register_full_backward_hook(self._get_grads_hook(name)))
            
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
            
    def forward(self, input):
        return self.net.forward(input)
        
    def __call__(self, input):
        return self.forward(input)

    def get_features(self, name):
        feats = [self.feature[name][k] for k in self.feature[name].keys()]
        if feats == []:
            return None
        else:
            return torch.cat(feats, dim=0)

    def get_grads(self, name):
        grads = [self.gradient[name][k] for k in self.gradient[name].keys()]
        if grads == []:
            return None
        else:
            return torch.cat(grads, dim=0)
    
    def reset_all(self):
        for layer in self.layer_name:
            self.feature[layer] = {}
            self.gradient[layer] = {}
            
    def get_all_features(self, c_reduce=None, hw_reduce=None):
        #<<CAUTION>> 
        # Assuming only BCHW for now! 
        # any other shape of feat&grad will produce weird results
        features_list = []
        max_size = torch.Tensor([0])
        min_size = torch.Tensor([torch.iinfo(torch.int64).max])
        for name in self.layer_name:
            feats = [self.feature[name][k] for k in self.feature[name].keys()]
            if feats == []:
                continue
            features_list.append(torch.cat(feats, dim=0))
            max_size = torch.max(max_size.expand_as(torch.Tensor(list(features_list[-1].shape[1:]))), 
                                 torch.Tensor(list(features_list[-1].shape[1:])))
            min_size = torch.min(min_size.expand_as(torch.Tensor(list(features_list[-1].shape[1:]))), 
                                 torch.Tensor(list(features_list[-1].shape[1:])))
        
        if c_reduce == None and hw_reduce == None:
            return features_list
        elif c_reduce == None or hw_reduce== None:
            raise ValueError("Currently cannot reduce on one direction. Do it manually")
        else:
            pass
        
        if c_reduce == 'all':
            #average all to 1-channel and add
            features_list = [torch.mean(f, dim=1, keepdims=True) for f in features_list]
        elif c_reduce == 'upscale':
            #interpolate shallower feats to deeper feats
            features_list = [torch.repeat_interleave(f, max_size[0].int().item()//f.shape[1], dim=1) for f in features_list]
        elif c_reduce == 'downscale':
            #interpolate deeper feats to shallower feats
            features_list = [F.interpolate(f.permute(0,2,3,1), (f.shape[2],min_size[0].int().item()), mode='nearest').permute(0,3,1,2) for f in features_list]
        else:
            raise ValueError

        if hw_reduce == 'upscale':
            #interpolate smaller feats to larger feats
            features_list = [F.interpolate(f, tuple(max_size[1:].int().numpy()), mode='bilinear') for f in features_list]
        elif hw_reduce == 'downscale':
            #interpolate larger feats to smaller feats
            features_list = [F.interpolate(f, tuple(min_size[1:].int().numpy()), mode='bilinear') for f in features_list]
        else:
            raise ValueError("")
        
        return torch.sum(torch.cat(features_list), dim=0,keepdims=True)

    def get_all_grads(self, c_reduce=None, hw_reduce=None):
        #<<CAUTION>> 
        # Assuming only BCHW for now! 
        # any other shape of feat&grad will produce weird results
        grads_list = []
        max_size = torch.Tensor([0])
        min_size = torch.Tensor([torch.iinfo(torch.int64).max])
        for name in self.layer_name:
            grads_list.append(torch.cat([self.gradient[name][k] for k in self.gradient[name].keys()], dim=0))
            max_size = torch.max(max_size.expand_as(torch.Tensor(list(grads_list[-1].shape[1:]))), 
                                 torch.Tensor(list(grads_list[-1].shape[1:])))
            min_size = torch.min(min_size.expand_as(torch.Tensor(list(grads_list[-1].shape[1:]))), 
                                 torch.Tensor(list(grads_list[-1].shape[1:])))
        
        if c_reduce == 'none' and hw_reduce == 'none':
            return grads_list
        elif c_reduce == 'none' or hw_reduce=='none':
            raise ValueError("Currently cannot reduce on one direction. Do it manually")
        else:
            pass
        
        if c_reduce == 'all':
            #average all to 1-channel and add
            grads_list = [torch.mean(g, dim=1, keepdims=True) for g in grads_list]
        elif c_reduce == 'upscale':
            #interpolate shallower feats to deeper feats
            grads_list = [torch.repeat_interleave(g, max_size[0].int().item()//g.shape[1], dim=1) for g in grads_list]
        elif c_reduce == 'downscale':
            #interpolate deeper feats to shallower feats
            grads_list = [F.interpolate(g.permute(0,2,3,1), (g.shape[2],min_size[0].int().item()), mode='nearest').permute(0,3,1,2) for g in grads_list]
        else:
            raise ValueError

        if hw_reduce == 'upscale':
            #interpolate smaller feats to larger feats
            grads_list = [F.interpolate(g, tuple(max_size[1:].int().numpy()), mode='bilinear') for g in grads_list]
        elif hw_reduce == 'downscale':
            #interpolate larger feats to smaller feats
            grads_list = [F.interpolate(g, tuple(min_size[1:].int().numpy()), mode='bilinear') for g in grads_list]
        else:
            raise ValueError("")
        
        return torch.sum(torch.cat(grads_list), dim=0,keepdims=True)
    
    def to(self, device):
        self.net.to(device)
        return self
    
    def train(self, mode=True):
        self.net.train(mode)
        return self
    
    def eval(self):
        self.net.eval()
        return self