import os
import argparse
import librosa
import soundfile
from tqdm import tqdm
import json

def is_overlap(a, b, min_gap=0.03):
    return (a[0] <= b[1] - min_gap) and (b[0] <= a[1] - min_gap)

def prepare_hflung(args):
    """Prepares HFLung V1 dataset into SADACO compatible format.
    https://gitlab.com/techsupportHF/HF_Lung_V1 

    Dataset should be unzipped following the hierarchy of
    
    data_root/
    └ train/
        └ steth_20180814_09_37_11_label.txt
        └ steth_20180814_09_37_11.wav
        └ ...
    └ test/
        └ steth_20190801_09_46_05_label.txt
        └ steth_20190801_09_46_05.wav
        └ ...
    
    Final result of the dataset preparation will look like
    
    data_root/
    └ train/
    └ test/
    └ sadaco/
        └ wavs/
            └ steth_20180814_09_37_11_001.wav
            └ ...
        └ meta.json
    
    Given 15 seconds length audio will be divided into cycles by default.
    Put --no_divide option if you want to keep the original 15seconds length data.
    """
    labels = {'Cycle':0, 'Multi' : 0, 'Normal': 0, 'Wheezes':0, 'Crackles':0, 'Rhonchi':0, 'Stridor':0}

    
    metadata = {'train':{'data':[], 'labels':[]}, 'test':{'data':[], 'labels':[]}}
    files = [('train', f) for f in os.listdir(os.path.join(args.data_root,'train'))] + [('test', f) for f in os.listdir(os.path.join(args.data_root,'test'))]
    wavs = [f for f in files if '.wav' in f[1]]
    txts = [f for f in files if '.txt' in f[1]]
    
    pbar = tqdm(enumerate(wavs), total=len(wavs), leave=False)
    for idx, wav_file in pbar:
        split, file = wav_file
        name = file.split('.')[0]
        file_path = os.path.join(args.data_root, split, file)
        ann_path = os.path.join(args.data_root, split, name+'_label.txt')
        out_path = os.path.join(args.data_root, 'sadaco', 'wavs')
        os.makedirs(out_path, exist_ok=True)
        waveform, _ = librosa.load(file_path, sr=args.sr, res_type='kaiser_fast')
        
        ann = open(ann_path,'r').readlines()
        cycles = {}
        symptoms = []
        cycle_idx = 0
        for a in ann:
            symp, start, end = a.replace('\n','').split(' ')
            start = list(map(float,start.split(':')))
            start = 3600*start[0]+60*start[1]+start[2]
            end = list(map(float,end.split(':')))
            end = 3600*end[0]+60*end[1]+end[2]
            if symp in ['I', 'E']:
                cycles[cycle_idx]= {}
                cycles[cycle_idx]['symps'] = []
                cycles[cycle_idx]['time'] = (start, end)
                cycle_idx += 1
            else:
                symptoms.append((symp, (start, end)))
        
        for symp in symptoms:
            sy, time = symp
            if sy == 'D':
                sy = 'Crackles'
            elif sy == 'Wheeze':
                sy = 'Wheezes'
            else:
                pass
            
            for cycle in cycles:
                if is_overlap(cycles[cycle]['time'], time):
                    cycles[cycle]['symps'].append(sy)
        
        pbar2 = tqdm(cycles.items(), total=len(cycles), leave=False)
        for sub_idx, cycle in pbar2:
            start, end = cycle['time']
            start, end = int(start*args.sr), int(end*args.sr)
            sy = cycle['symps']
            if sy == []:
                sy.append('Normal')
            
            chunk = waveform[start:end+1]
            save_path = os.path.join(out_path, f'{name}_{sub_idx:04}.wav')
            soundfile.write(save_path, chunk, samplerate=args.sr)
            metadata[split]['data'].append(save_path)
            cycle_label = sy
            metadata[split]['labels'].append(cycle_label)
            
            for lab in sy:
                labels[lab] += 1
            labels['Cycle'] += 1
            if len(sy) >= 2:
                labels['Multi'] +=1 
        
            pbar.set_postfix(labels)
            
    json.dump(metadata, open(os.path.join(args.data_root,'sadaco', 'meta.json'), 'w'))
        
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--sr', type=int, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    prepare_hflung(args)