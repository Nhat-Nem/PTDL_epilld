import sys
sys.path.append("src")

import torch
from models.embedding_model import EmbeddingModel
from models.multihead_model import MultiheadModel

n_classes = 9804

E_model = EmbeddingModel(
    network="resnet18",
    pooling="CBP",
    dropout_p=0.5,
    cont_dims=2048,
    pretrained=False
)

model = MultiheadModel(E_model, n_classes, train_with_side_labels=False)

state = torch.load("pill_model.pth", map_location="cpu")

if "model_state_dict" in state:
    state = state["model_state_dict"]

model.load_state_dict(state)

model.eval()

print("Model loaded successfully")