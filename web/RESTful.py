import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


IMG_EXT = ['png', 'jpg', 'tif', 'tiff', 'jpeg', 'bmp']

FAIL_INTERNAL = 'Internal Process Failure : '

app = FastAPI()
app.mount("/", StaticFiles(directory="asset"), name="static")

def isimage(filename):
    extension = filename.split('.')[-1]
    return (extension in IMG_EXT)

def get_api(filename, root='image'):
    filename = '.'.join(filename.split('.')[:-1])
    annot_path = os.path.join(root, filename + '.txt')
    if os.path.isfile(annot_path):
        annot = open(annot_path).read()
        annot = [int(a) for a in annot.split(' ')]
    else:
        annot = False
    return annot

def api_test(image_id, api_numbers):
    if not os.path.isfile('files.json'):
        get_images()
    info = json.load(open('files.json', 'rb'))
    if not isinstance(api_numbers, list):
        api_numbers = [api_numbers]
    return image_id in info and all([info[image_id]['api'][i] for i in api_numbers])

@app.get('/images')
def get_images():
    file_list = os.listdir('image')
    image_list = sorted([i for i in file_list if isimage(i) and get_api(i)])
    info = {name: {'id':name, 'url':f'image/{name}', 'api': get_api(name)} for idx, name in enumerate(image_list)}
    json.dump(info, open('files.json', 'w'))
    return list(info.values())


@app.get('/bbox/{image_id}')
def run_sia(image_id:str):
    if api_test(image_id, 0):
        return FileResponse('sia.json')
    return "Input Error"

@app.get('/attr_temporal/{image_id}/{obj_id}')
def run_ych(image_id: str, obj_id:int):
    import random
    if not api_test(image_id, 4):
        return "Input Error"
 
    # TODO
    obj_id = 0
    # override to actually use obj_id  
    
    import numpy as np
    def softmax(x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(y)
        return f_x
    cls_score = np.random.randn(10)
    idx2cls = ['MaritimeVessel', 'Motorboat', 'Sailboat', 'Tugboat', 'Barge', 'FishingVessel', 'Ferry', 'Yacht', 'ContainerShip', 'OilTanker']
    
    hardcoded = {
    'original':{
        'obj_img': 'image/4002.png', 
        'obj_name' : idx2cls[np.argmax(cls_score)],
        'orig_scores': softmax(cls_score).tolist(),
        'idx2cls': idx2cls, 
    },
    'attr':{
        'attr_result' : [True if random.random()>0.85 else False for i in range(13)],
        'attr_idx2cls': ['launcher', 'motor engine', 'sail', 'tire fender', 'storage', 'crane', 'pool', 'layered floor', 'container', 'dome', 'command post', 'chimmy', 'ship head'], 
        'attr_scores': softmax(np.random.randn(10)).tolist(),
    },
    'temporal':{
        'temporal_img': 'image/4003.png', 
        'temporal_scores':softmax(np.random.randn(10)).tolist(),
    },
    'modified_scores':softmax(np.random.randn(10)).tolist(),

    'attr_cls2idx': {'launcher': 0, 'motor engine': 1, 'sail': 2, 'tire fender': 3, 'storage': 4, 'crane': 5, 'pool': 6, 'layered floor': 7, 'container': 8, 'dome': 9, 'command post':10, 'chimmy':11, 'ship head':12}, 
    'cls2idx': {'MaritimeVessel': 0, 'Motorboat': 1, 'Sailboat': 2, 'Tugboat': 3, 'Barge': 4, 'FishingVessel': 5, 'Ferry': 6, 'Yacht': 7, 'ContainerShip': 8, 'OilTanker': 9}, 
    }

    return hardcoded

@app.get('/attr_graph/{image_id}/{obj_id}')
def run_kci(image_id:str, obj_id: int):
    if not api_test(image_id, 3):
        return "Input Error"

    return FileResponse('kci.json')



