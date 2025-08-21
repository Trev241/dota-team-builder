from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import torch
import torch.nn as nn
import json

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ICON_URL = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/"
heroes = []
for h in json.load(open("assets/heroes.json"))["heroes"]:
    h["icon_url"] = ICON_URL + h["name"] + ".png"
    heroes.append(h)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, attn_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attn_dim)
        self.key = nn.Linear(input_dim, attn_dim)
        self.value = nn.Linear(input_dim, attn_dim)
        self.scale = attn_dim**0.5

    def forward(self, x):
        # x shape: [batch, N, input_dim] where N = 4 (heroes)
        Q = self.query(x)  # [batch, N, attn_dim]
        K = self.key(x)  # [batch, N, attn_dim]
        V = self.value(x)  # [batch, N, attn_dim]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, N, N]
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, N, N]
        out = torch.matmul(attn_weights, V)  # [batch, N, attn_dim]
        return out, attn_weights  # Optionally return attn_weights for inspection


# Define your model class (simplified placeholder)
class HeroPredictor(nn.Module):
    def __init__(
        self,
        num_heroes=113,
        embed_dim=64,
        num_roles=5,
        role_embed_dim=16,
        hidden_dim1=512,
        hidden_dim2=256,
        attn_dim=64,
    ):
        super(HeroPredictor, self).__init__()
        self.embedding = nn.Embedding(num_heroes, embed_dim)
        self.role_embedding = nn.Embedding(num_roles, role_embed_dim)

        # input_dim = (embed_dim + role_embed_dim) * 4

        self.input_dim = embed_dim + role_embed_dim
        self.attn = SelfAttention(self.input_dim, attn_dim)

        self.fc1 = nn.Linear(attn_dim * 4, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_heroes)
        self.relu = nn.LeakyReLU(0.1)  # Better gradient flow

    def forward(self, hero_ids, role_ids):
        sorted_heroes, indices = torch.sort(hero_ids, dim=1)
        sorted_roles = torch.gather(role_ids, dim=1, index=indices)

        hero_embed = self.embedding(sorted_heroes)  # [batch, 4, embed_dim]
        role_embed = self.role_embedding(sorted_roles)  # [batch, 4, role_embed_dim]

        combined = torch.cat(
            [hero_embed, role_embed], dim=-1
        )  # [batch, 4, embed + role]
        # combined_flat = combined.view(combined.size(0), -1)

        attn_out, attn_weights = self.attn(combined)  # [batch, 4, attn_dim]
        attn_out_flattened = attn_out.view(attn_out.size(0), -1)  # [batch, 4*attn_dim]

        # flattened = embedded.view(embedded.size(0), -1)
        # pooled = embedded.mean(dim=1)  # [batch_size, embed_dim] — order-invariant

        out = self.fc1(attn_out_flattened)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)  # [batch_size, num_heroes]

        return out, attn_weights


# Load your pre-trained model weights here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HeroPredictor(
    num_heroes=113,
    embed_dim=128,
    num_roles=5,
    role_embed_dim=32,
    hidden_dim1=1024,
    hidden_dim2=512,
    attn_dim=128,
)
checkpoint = torch.load("assets/best_model.pth", map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)


# Pydantic schema for input
class HeroInput(BaseModel):
    heroes: List[int]  # List of 4 hero IDs as input


# API endpoint to predict next hero
@app.post("/predict")
def predict(input: HeroInput):
    if len(input.heroes) != 4:
        return {"error": "Exactly 4 hero IDs are required as input."}

    # Convert input to tensor
    hero_tensor = torch.tensor([input.heroes], dtype=torch.long)  # Batch size 1
    with torch.no_grad():
        logits = model(hero_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    # Get top 5 predictions (hero IDs and probabilities)
    top_probs, top_indices = torch.topk(probs, 5)
    predictions = [
        {"hero_id": int(idx), "probability": float(prob)}
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {"predictions": predictions}


@app.get("/heroes")
def get_heroes():
    return heroes
