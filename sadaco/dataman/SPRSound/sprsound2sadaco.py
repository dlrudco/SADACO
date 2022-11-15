import os
import argparse
import librosa
import soundfile
from tqdm import tqdm
import json

def is_overlap(a, b, min_gap=0.03):
    return (a[0] <= b[1] - min_gap) and (b[0] <= a[1] - min_gap)

def prepare_sprsound(args):
    """Prepares HFLung V1 dataset into SADACO compatible format.
    https://gitlab.com/techsupportHF/HF_Lung_V1 

    Dataset should be unzipped following the hierarchy of
    
    data_root/
    └ train_wav/
        └ 41074281_5.0_1_p4_408.wav
        └ ...
    └ test_wav/
        └ 41087486_3.8_0_p4_232.wav
        └ ...
    └ train_json/
        └ 41074281_5.0_1_p4_408.json
        └ ...
    └ test_json/
        └ 41087486_3.8_0_p4_232.json
        └ ...
        
    Final result of the dataset preparation will look like
    
    data_root/
    └ train_wav/
    └ test_wav/
    └ train_json/
    └ test_json/
    └ sadaco/
        └ wavs/
            └ 41074281_5.0_1_p4_408_001.wav
            └ ...
        └ meta.json
    """
    '''
    'r': {
        'Normal': True, 'DAS': True, 'CAS': True, 
        'CAS & DAS': True, 'Poor Quality': True
        } 
    'e': {
        'Normal': True, 'Fine Crackle': True, 'Rhonchi': True, 
        'Wheeze': True, 'Coarse Crackle': True, 'Stridor': True, 
        'Wheeze+Crackle': True}
    '''
    labels = {'Cycle':0, 'Multi' : 0, 'Normal': 0, 'Wheezes':0, 'Crackles':0, 'Rhonchi':0, 'Stridor':0}
    
    
    metadata = {'train':{'data':[], 'labels':[], 'labels_coarse':[]}, 'test':{'data':[], 'labels':[], 'labels_coarse':[]}}
    files = [('train', f) for f in os.listdir(os.path.join(args.data_root,'train_wav'))] + [('test', f) for f in os.listdir(os.path.join(args.data_root,'test_wav'))]
    
    pbar = tqdm(enumerate(files), total=len(files), leave=False)
    for idx, wav_file in pbar:
        split, file = wav_file
        name = '.'.join(file.split('.')[:-1])
        file_path = os.path.join(args.data_root, split+'_wav', file)
        ann_path = os.path.join(args.data_root, split+'_json', name+'.json')
        out_path = os.path.join(args.data_root, 'sadaco', 'wavs')
        os.makedirs(out_path, exist_ok=True)
        waveform, _ = librosa.load(file_path, sr=args.sr, res_type='kaiser_fast')
        
        ann = json.load(open(ann_path,'r'))
        '''
        Below code snippet is only for checking if there's overlapping events.
        Uncomment if thought suspicious.
        '''
        # cycles = {}
        # cycle_idx = 0
        # for a in ann['event_annotation']:
        #     start, end, symp = a['start'], a['end'], a['type']
        #     start = 1e-3 * int(start)
        #     end = 1e-3 * int(end)
        #     if list(cycles) == []:
        #         cycles[cycle_idx]= {}
        #         cycles[cycle_idx]['symps'] = [symp]
        #         cycles[cycle_idx]['time'] = (start, end)
        #         cycle_idx += 1
        #     else:
        #         flag = True
        #         for k, cycle in cycles.items():
        #             if is_overlap(cycle['time'], (start, end)):
        #                 print('hello')
        #                 cycles[k]['symps'].append(symp)
        #                 flag=False
        #         if flag:
        #             cycles[cycle_idx]= {}
        #             cycles[cycle_idx]['symps'] = [symp]
        #             cycles[cycle_idx]['time'] = (start, end)
        #             cycle_idx += 1
        rean = ann['record_annotation']
        events = ann['event_annotation']
        pbar2 = tqdm(enumerate(events), total=len(events), leave=False)
        for sub_idx, cycle in pbar2:
            start, end, symp = cycle['start'], cycle['end'], cycle['type']
            start = 1e-3 * int(start)
            end = 1e-3 * int(end)
            start, end = int(start*args.sr), int(end*args.sr)

            chunk = waveform[start:end+1]
            save_path = os.path.join(out_path, f'{name}_{sub_idx:04}.wav')
            soundfile.write(save_path, chunk, samplerate=args.sr)
            metadata[split]['data'].append(save_path)
            if symp == 'Wheeze+Crackle':
                symp = ['Wheezes', 'Crackles']
            elif symp == 'Wheeze':
                symp = ['Wheezes']
            elif symp in ['Fine Crackle', 'Coarse Crackle']:
                symp = ['Crackles']
            else:
                symp = [symp]
                
            cycle_label = symp
            metadata[split]['labels'].append(cycle_label)
            metadata[split]['labels_coarse'].append(rean)
            
            for lab in symp:
                labels[lab] += 1
            labels['Cycle'] += 1
            if len(symp) >= 2:
                labels['Multi'] +=1 
        
            pbar.set_postfix(labels)
    print(labels)
    json.dump(metadata, open(os.path.join(args.data_root,'sadaco', 'meta.json'), 'w'))
        
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--sr', type=int, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = args_parser()
    prepare_sprsound(args)