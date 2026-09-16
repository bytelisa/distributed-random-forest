import yaml
import os

# Reads the configuration file for the Worker

class Config:
    def __init__(self, config_path="configs/config.yaml"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, 'r') as file:
            self._cfg = yaml.safe_load(file)

    @property
    def storage_endpoint(self):
        return self._cfg['storage']['endpoint']

    @property
    def storage_access_key(self):
        return self._cfg['storage'].get('access_key')

    @property
    def storage_secret_key(self):
        return self._cfg['storage'].get('secret_key')

    @property
    def storage_bucket(self):
        return self._cfg['storage']['bucket']

    @property
    def local_temp_dir(self):
        return self._cfg['storage'].get('local_temp_dir', 'temp_data')

    @property
    def grpc_max_threads(self):
        # Default to 10 if not found
        return self._cfg.get('workers', {}).get('grpc_max_threads', 10)

    @property
    def model_prefix(self):
        # Default to "models/"
        return self._cfg['storage'].get('base_path_prefix', 'models/')

    def model_defaults(self, task_type: str) -> dict:
        # Hyperparameters applied when a request doesn't specify them, per task
        return dict(self._cfg['model_defaults'][task_type])

# Global instance or factory to load it
def load_config():
    return Config()