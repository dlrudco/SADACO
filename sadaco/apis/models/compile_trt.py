from sklearn import model_selection
import torch
import time
from torch2trt import torch2trt

def compile(model : torch.nn.Module, input_shape : tuple = (1,3,224,224),
            device_id=0, batch_size=1, checkpoint=None, output_names=None):
    if torch.cuda.is_available() is False:
        print('No CUDA Found Available! Aborting TRT Compile')
        return model, None
    else:
        if output_names is None:
            output_names = ['out']
        st = time.time()
        print('Converting...')
        model_trt = torch2trt(model, [torch.randn(input_shape).cuda(device_id)],
            fp16_mode=True, use_onnx=True, output_names = output_names, max_batch_size=batch_size)
        print("Done")
        print(time.time()-st)
        return model, model_trt
    

    