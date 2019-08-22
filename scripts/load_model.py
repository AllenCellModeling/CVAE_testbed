import torch 
import sys
sys.path.insert(0, '../.')
from models.model_loader import ModelLoader

path = "../outputs/baseline_results2/2019:08:15:08:22:25/"

this_model_instance = ModelLoader(None, path)
proj_matrix = this_model_instance.load_projection_matrix(None)
print(proj_matrix)
weights = this_model_instance.load_model(None)


