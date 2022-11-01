import os
import argparse
import librosa
import soundfile
from tqdm import tqdm
import json

def is_overlap(a, b, min_gap=0.03):
    return (a[0] <= b[1] - min_gap) and (b[0] <= a[1] - min_gap)

def prepare_icbhi(args):
    """Prepares ICBHI dataset into SADACO compatible format.
    https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge 

    ICBHI dataset should be unzipped following the hierarchy of
    
    data_root/
    └ wavs/
        └ 101_1b1_Al_sc_Meditron.wav
        └ ...
    └ txt/
        └ 101_1b1_Al_sc_Meditron.txt
        └ ...
    └(Optional) event_txt/
        └ 101_1b1_Al_sc_Meditron_events.txt
        └ ...
    └ ICBHI_Challenge_diagnosis.txt
    └ ICBHI_Challenge_train_test.txt
    
    Final result of the dataset preparation will look like
    
    data_root/
    └ wavs/
    └ txt/
    └ event_txt/
    └ ICBHI_Challenge_diagnosis.txt
    └ ICBHI_Challenge_train_test.txt
    └ sadaco/
        └ wavs/
            └ 101_1b1_Al_sc_Meditron_01.wav
            └ ...
        └ meta.json
    """
    labels = ['Normal', 'Wheezes', 'Crackles', 'Crackles&Wheezes']
    labels2 = [['Normal'], ['Wheezes'], ['Crackles'], ['Crackles','Wheezes']]
    lc = [0] * 4
    dc = {}
    metadata = {'train':{'data':[], 'labels':[], 'labels_diagnosis':[]}, 'test':{'data':[], 'labels':[], 'labels_diagnosis':[]}}
    wav_path = os.path.join(args.data_root, 'wavs')
    txt_path = os.path.join(args.data_root, 'txt')
    wavs = sorted(os.listdir(wav_path))
    try:
        splits = {sp.replace('\n', '').split('\t')[0]:sp.replace('\n', '').split('\t')[1] for sp in open(os.path.join(args.data_root, 'ICBHI_challenge_train_test.txt')).readlines()}
    except IndexError:
        print(open(os.path.join(args.data_root, 'ICBHI_challenge_train_test.txt')).readlines())
        breakpoint()
    try:
        diagnosis = {dg.replace('\n', '').split('\t')[0]:dg.replace('\n', '').split('\t')[1] for dg in open(os.path.join(args.data_root, 'ICBHI_Challenge_diagnosis.txt')).readlines()}
    except IndexError:
        print(open(os.path.join(args.data_root, 'ICBHI_Challenge_diagnosis.txt')).readlines())
        breakpoint()
    for pid, diags in diagnosis.items():
        dc[diags] = 0  
    pbar = tqdm(enumerate(wavs), total=len(wavs), leave=False)
    for idx, wav in pbar:
        name = wav.split('.')[0]
        if name not in splits.keys():
            print(name)
            continue
        pid = name.split('_')[0]
        file_path = os.path.join(wav_path, wav)
        out_path = os.path.join(args.data_root, 'sadaco', 'wavs')
        os.makedirs(out_path, exist_ok=True)
        waveform, _ = librosa.load(file_path, sr=args.sr, res_type='kaiser_fast')
        cycles = [list(map(float,c.replace('\n', '').split('\t'))) for c in open(os.path.join(txt_path, name+'.txt')).readlines()]
        pbar2 = tqdm(enumerate(cycles), total=len(cycles), leave=False)
        for sub_idx, cycle in pbar2:
            sp = splits[name]
            dg = diagnosis[pid] 
            start, end, crackle, wheeze = cycle
            start, end = int(start*args.sr), int(end*args.sr)
            crackle, wheeze = int(crackle), int(wheeze)
            chunk = waveform[start:end+1]
            save_path = os.path.join(out_path, f'{name}_{sub_idx:04}.wav')
            soundfile.write(save_path, chunk, samplerate=args.sr)
            metadata[sp]['data'].append(save_path)
            cycle_label = labels2[int(f'{crackle}{wheeze}', 2)]
            metadata[sp]['labels'].append(cycle_label)
            metadata[sp]['labels_diagnosis'].append(dg)
            lc[int(f'{crackle}{wheeze}', 2)] += 1
            dc[dg] += 1
            pbar.set_postfix({l:c for l,c in zip(labels, lc)})
            pbar2.set_postfix(dc)
    print({l:c for l,c in zip(labels, lc)})
    json.dump(metadata, open(os.path.join(args.data_root,'sadaco', 'meta.json'), 'w'))
        
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--sr', type=int, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    prepare_icbhi(args)