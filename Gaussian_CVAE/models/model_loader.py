import torch
import pathlib
from pathlib import Path
from torch import nn
import logging

LOGGER = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model: nn.Module, path_save_dir: Path) -> None:
        """
        Args: 
        trained model: weights of trained neural network
        path_save_dir: Path of saved model    
        """
        self.model = model
        self.path_save_dir = path_save_dir

    def save_model(self):
        """Saves model weights and metadata in specified directory."""
        self.path_save_dir.mkdir(parents=True, exist_ok=True)
        path_weights = self.path_save_dir / Path('weights.pt')
        device = next(self.model.parameters()).device  # Get device from first param
        self.model.to(torch.device('cpu'))
        torch.save(self.model.state_dict(), path_weights)
        LOGGER.info(f'Saved model weights: {path_weights}')
        self.model.to(device)
    
    def load_model(self, path_save_dir):
        """Loads model weights from specified directory"""
        if path_save_dir is not None:
            path_weights = path_save_dir / Path('weights.pt')
        else:
            path_weights = self.path_save_dir / Path('weights.pt')
        model = torch.load(path_weights)
        model.eval()
        return model
    
    def load_projection_matrix(self, path_save_dir):
        """Loads model weights from specified directory"""
        if path_save_dir is not None:
            path_weights = path_save_dir / Path('projection_options.pt')
        else:
            path_weights = self.path_save_dir / Path('projection_options.pt')
        projection_matrix = torch.load(path_weights)
        return projection_matrix