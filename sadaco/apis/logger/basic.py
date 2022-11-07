import os
import json
from datetime import datetime
from typing import DefaultDict
import string
import random

class BaseLogger:
    def __init__(self, config, log_path = 'logs'):
        self.configs = config
        self.group_id = self.configs['Master']['prefix']
        self.now = datetime.now()
        self.name =  self.generate_expid()
        self.log_path = os.path.join(log_path, self.group_id, self.name)
        os.makedirs(self.log_path, exist_ok=True)
        self.exp_logs = os.path.join(self.log_path, 'log.txt')
        json.dump(self.configs, open(os.path.join(self.log_path, 'configs.json'), 'w'))
        
    def log(self, logs : DefaultDict):
        logstring = f"{self.now.strftime('%H:%M:%S')} : "
        logstring += ''.join([f'{k} : {v}\t' for k,v in logs.items()])
        logstring += '\n'
        with open(self.exp_logs, 'a') as el:
            el.write(logstring)
            
    def generate_expid(self, size=6, chars=string.ascii_lowercase+string.digits):
        return self.now.strftime('%Y%m%d_%H%M%S_') + ''.join(random.choice(chars) for _ in range(size))
