import argparse
import json

############################## Parsing Arguments ##############################
class Parser:
    def __init__(self):
        self.parameters = {}

    def parse_arguments(self, coord_params):
        parser = argparse.ArgumentParser()

        default_port = None
        if coord_params != "":
            json_vars = json.loads(coord_params)
            self.parameters.update(json_vars)
            if 'port' in self.parameters:
                default_port = self.parameters['port']

        parser.add_argument('--port', type=int, required=False, default = default_port, help="port to run the gRPC server on")
        flag_params = parser.parse_args()
        self.parameters.update(vars(flag_params))
    
        return self.parameters