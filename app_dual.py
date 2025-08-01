
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from model import DPLSTMBlock

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo use, can be locked down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
highres_model = DPLSTMBlock(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)
checkpoint = torch.load("dp_model_real.npz", map_location=device)

# Handle DataParallel key mismatch
if any(k.startswith("_module.") for k in checkpoint.keys()):
    checkpoint = {k.replace("_module.", ""): v for k, v in checkpoint.items()}

highres_model.load_state_dict(checkpoint)
highres_model.eval()


class GenRequest(BaseModel):
    num_samples: int = 100
    seq_len: int = 1250


@app.post("/generate")
def generate(req: GenRequest):
    with torch.no_grad():
        noise = torch.randn(req.num_samples, req.seq_len, 2).to(device)
        out = highres_model(noise).cpu().numpy()
    return {"samples": out.tolist()}


@app.get("/")
def root():
    return {"message": "VitalSynth backend is live"}
