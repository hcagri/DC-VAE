from lib import *
import os 

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred import SETTINGS
SETTINGS.CONFIG.READ_ONLY_CONFIG=False

experiment_dir = "/Users/hcagri/Documents/METU-Master/Term II/CENG 796/project/Dual-Contradistinctive-Generative-Autoencoder/runs"

ex = Experiment("ceng796")
ex.observers.append(FileStorageObserver(experiment_dir))
ex.add_config(configs)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)
    
    os.makedirs(os.path.join(experiment_dir, _run._id, "checkpoints"))
    os.makedirs(os.path.join(experiment_dir, _run._id, "results"))
    
    checkpoint = torch.load('checkpoint_7500.pt', map_location = _config['hparams']['device'])

    train(_config['model_params'], _config['hparams'], _run, checkpoint)
