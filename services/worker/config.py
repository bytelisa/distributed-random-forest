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
        return self._cfg['storage']['access_key']

    @property
    def storage_secret_key(self):
        return self._cfg['storage']['secret_key']

    @property
    def storage_bucket(self):
        return self._cfg['storage']['bucket']

    @property
    def local_temp_dir(self):
        return self._cfg['storage'].get('local_temp_dir', 'temp_data')

# Global instance or factory to load it
def load_config():
    return Config()