import time
import torch
from tqdm import tqdm
from apis.models import custom, compile_trt

if __name__ == '__main__':
    import torch
import torchaudio
from PIL import Image
from utils.config_parser import parse_config_obj
from apis.models import build_model
master_config = '../demo_materials/demo_configs.yml'
model_config = '../demo_materials/demo_model.yml'
input_path = '../demo_materials/wheeze.wav'
#180_1b4_Al_mc_AKGC417L_2_11.wav
model_checkpoint = '../demo_materials/demo_ckp.pth'

if __name__ == "__main__":
    
    master_cfg = parse_config_obj(master_config)
    model_cfg = parse_config_obj(model_config)
    
    model = build_model(model_cfg)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_checkpoint)

    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module
    
    #TODO:merge below pipeline into unified apis
    waveform, _ = torchaudio.load(input_path)
    waveform = waveform - waveform.mean()
    
    cart = torch.stft(waveform, n_fft = int(1e-3*70*16000+1), 
                           hop_length=int(1e-3*25*16000),
                           window = torch.hann_window(int(1e-3*70*16000+1))
                           )
    phase = torch.atan2(cart[:,:,:,1], cart[:,:,:,0])
    mag = cart[:,:,:,0]**2 + cart[...,1]**2
    if mag.shape[-1] < 128:
        mag = mag.repeat(1, 1, 128//mag.shape[-1] + 1)
        mag = mag[:,:,:128]
    else:
        mag = mag[:,:,:128]
    melscale = torchaudio.transforms.MelScale(sample_rate=16000, n_mels=128, n_stft=mag.shape[1]).cuda()
    inputs = melscale(mag.cuda().float())
    
    norm_mean = -4.2677393
    norm_std = 4.5689974
    inputs = (inputs - norm_mean) / (norm_std * 2)
    #
    
    model, model_trt = compile_trt.compile(model, input_shape=(1,1,128,128))
    model_trt = model_trt.cuda()
    original_model_time = 0
    accel_model_time = 0
    for i in range(10):
        inp = torch.randn((1,1,128,128)).cuda()
        _ = model(inp)
    for i in tqdm(range(100)):
        inp = torch.randn((1,1,128,128)).cuda()
        st = time.time()
        _ = model(inp)
        original_model_time += (time.time()-st)*10
    print(f"\nOriginal Model Execution Time(Avg.): {original_model_time:.2f}ms\n")
    for i in range(10):
        inp = torch.randn((1,1,128,128)).cuda()
        _ = model_trt(inp)
    for i in tqdm(range(100)):
        inp = torch.randn((1,1,128,128)).cuda()
        st = time.time()
        _ = model_trt(inp)
        accel_model_time += (time.time()-st)*10
    print(f"\nAccelerated Model Execution Time(Avg.): {accel_model_time:.2f}ms\n")
    
    outputs = model(inputs.unsqueeze(0))
    outputs = torch.softmax(outputs/outputs.abs().max(), dim=1)[0]
    print("Results of Classifying wheeze.wav")
    print("Original")
    print(f"\tPred : [ Normal {outputs[0]*100:.2f} %, Wheeze {outputs[1]*100:.2f} %, Crackle {outputs[2]*100:.2f} %, Both {outputs[3]*100:.2f} % ]") 
    
    outputs = model_trt(inputs.unsqueeze(0))
    outputs = torch.softmax(outputs/outputs.abs().max(), dim=1)[0]
    print("TensorRT FP16 model")
    print(f"\tPred : [ Normal {outputs[0]*100:.2f} %, Wheeze {outputs[1]*100:.2f} %, Crackle {outputs[2]*100:.2f} %, Both {outputs[3]*100:.2f} % ]")
    