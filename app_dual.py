
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from torch import nn
from vitalsynth_edgecases import apply_edge_cases

app = FastAPI()

# LSTM for high-res
class DPLSTMBlock(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.linear(h)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load high-resolution model
highres_model = DPLSTMBlock().to(device)
highres_model.load_state_dict(torch.load("dp_model_real.npz", map_location=device))
highres_model.eval()

@app.get("/")
def root():
    return {"message": "VitalSynth API - Dual Mode (low + high res)"}

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.1.0", "modes": ["40-step", "1250-step"]}

class GenRequest(BaseModel):
    num_samples: int = 10
    edge_cases: bool = False

@app.post("/generate_lowres")
def generate_lowres(req: GenRequest):
    output = []
    for _ in range(req.num_samples):
        base = np.random.normal(0, 1, (40, 2))
        if req.edge_cases:
            sample = apply_edge_cases(base, afib=True, dropout=True, motion=True)
        else:
            sample = base
        output.append(sample.tolist())
    return {"data": output}

@app.post("/generate_highres")
def generate_highres(req: GenRequest):
    output = []
    for _ in range(req.num_samples):
        z = torch.randn(1, 1250, 2).to(device)
        with torch.no_grad():
            x_hat = highres_model(z).cpu().numpy()[0]
        if req.edge_cases:
            x_hat = apply_edge_cases(x_hat, afib=True, dropout=True, motion=True)
        output.append(x_hat.tolist())
    return {"data": output}
