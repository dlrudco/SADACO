import os
import argparse
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Union
from munch import DefaultMunch
import yaml

def parse_config_dict(
    yml_path: str = None,
    arg_type: str = "data",
) -> Dict :
    '''
    Take a yaml file and return the corresponding arguments.

    Args:
        arg_type (str): The type of arguments to return. One of "data", "frontend", "model", "train". (default: "data")

    Returns:
        Dict: The corresponding arguments in dictionary form.
    '''
    
    if yml_path is None:
        yml_file = os.path.join(os.getcwd(), "configs_", arg_type + ".yml")
    else:
        yml_file = yml_path

    with open(yml_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def parse_config_obj(yml_path: str = None):
    with open(yml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        obj = DefaultMunch.fromDict(config)
    return obj

class ArgsParser():
    def __init__(self, argv=None):
        # Do argv default this way, as doing it in the functional
        # declaration sets it at compile time.
        if argv is None:
            self.argv = sys.argv
        else:
            self.argv = argv
        
        # Parse any conf_file specification
        # We make this parser with add_help=False so that
        # it doesn't parse -h and print help.
        self.default_parser = argparse.ArgumentParser(
            description=__doc__,  # printed with -h/--help
            # Don't mess with format of description
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # Turn off help, so we print all options in response to -h
            add_help=False,
        )
        self.default_parser.add_argument(
            "-c", "--conf_file", help="Specify the master config file", metavar="FILE", required=False,
            default = None
        )
        self.default_args, self.remaining_argv = self.default_parser.parse_known_args()
        self.parser = argparse.ArgumentParser(
            parents=[self.default_parser]
            )
        if self.default_args.conf_file is not None:
            configs = parse_config_obj(yml_path=self.default_args.conf_file)
            self.parser.set_defaults(**configs.__dict__)
        else:
            pass
        
    def add_argument(self, *opts, **kwopts):
        self.parser.add_argument(*opts, **kwopts)
        
    def get_args(self):
        args = self.parser.parse_args(self.remaining_argv)
        return args
        
    # def ParseArgswithConfig(argv=None):
        
        
        
        

    #     # Parse rest of arguments
    #     # Don't suppress add_help here so it will handle -h
    #     parser = argparse.ArgumentParser(
    #         # Inherit options from config_parser
    #         parents=[conf_parser]
    #     )
    #     parser.set_defaults(**configs.__dict__)

    #     parser.add_argument('--seed')
    #     parser.add_argument('--gpus')
    #     parser.add_argument('--model_configs')
    #     parser.add_argument('--data_configs')

        
    #     return args


if __name__ == "__main__":
    my_parser = ArgsParser()
    print(my_parser.default_args)
    my_parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
    my_parser.add_argument("--mixup", action='store_true')
    args = my_parser.get_args()
    print(args)
    breakpoint()