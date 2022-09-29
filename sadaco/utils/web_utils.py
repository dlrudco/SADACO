import os
from sadaco import pipelines
from sadaco.utils import config_parser

locate_configs = pipelines.__spec__.submodule_search_locations

def get_configs():        
    paths = []
    for dir in locate_configs:
        if os.path.isdir(os.path.join(dir,'configs')):
            paths.extend(os.listdir(os.path.join(dir,'configs')))
    paths = [p for p in paths if ('.yml' in p) or ('.yaml' in p)]
    return paths

def load_config(name):
    ap = None
    for loc in locate_configs:
        file_path = os.path.join(loc, 'configs', name)
        if os.path.isfile(file_path):
            ap = config_parser.parse_config_obj(yml_path=file_path)
            location = loc
            break
    print(ap, location)
    ap.data_configs.file = os.path.join(location,ap.data_configs.file)
    ap.model_configs.file = os.path.join(location,ap.model_configs.file)
    return ap