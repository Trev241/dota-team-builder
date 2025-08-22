from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from roles import role_map

import torch
import torch.nn as nn
import torch.nn.functional as F
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
heroes_lookup = {h["id"]: h for h in heroes}


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


def fill_heroes_with_similar(heroes_list, embedding_layer, num_total=4):
    n = len(heroes_list)
    if n == 0:
        raise ValueError("At least one hero ID required")

    if n >= num_total:
        return heroes_list[:num_total]

    heroes_tensor = torch.tensor(heroes_list, dtype=torch.long)
    chosen_embeds = embedding_layer(heroes_tensor)
    avg_embed = chosen_embeds.mean(dim=0, keepdim=True)
    all_embeds = embedding_layer.weight

    avg_embed_norm = F.normalize(avg_embed, p=2, dim=1)
    all_embeds_norm = F.normalize(all_embeds, p=2, dim=1)
    similarities = torch.matmul(all_embeds_norm, avg_embed_norm.t()).squeeze(1)

    mask = torch.ones_like(similarities, dtype=torch.bool)
    mask[heroes_tensor] = False
    similarities = similarities.masked_fill(~mask, float("-inf"))

    num_to_pick = num_total - n
    _, top_indices = torch.topk(similarities, num_to_pick)

    filled_heroes = heroes_list + top_indices.tolist()
    return filled_heroes


@app.post("/predict")
def predict_next_hero_order(input: HeroInput):
    heroes_list = input.heroes
    n = len(heroes_list)

    if n < 1 or n > 4:
        return {"error": "Provide between 1 and 4 hero IDs."}

    heroes_list = [h - 1 for h in heroes_list]  # to 0-based

    # Compute cosine similarity scores of all heroes vs avg embedding of chosen heroes
    def cosine_similarity_scores(chosen_heroes):
        chosen_tensor = torch.tensor(chosen_heroes, dtype=torch.long)
        chosen_embeds = model.embedding(chosen_tensor)  # [n, embed_dim]
        avg_embed = chosen_embeds.mean(dim=0, keepdim=True)  # [1, embed_dim]

        all_embeds = model.embedding.weight  # [num_heroes, embed_dim]

        avg_embed_norm = F.normalize(avg_embed, p=2, dim=1)  # [1, embed_dim]
        all_embeds_norm = F.normalize(all_embeds, p=2, dim=1)  # [num_heroes, embed_dim]

        sims = torch.matmul(all_embeds_norm, avg_embed_norm.t()).squeeze(
            1
        )  # [num_heroes]

        return sims

    # Predict next hero using cosine similarity for slots 2-4
    if n < 4:
        sims = cosine_similarity_scores(heroes_list)
        chosen_set = set(heroes_list)
        # Sort heroes by similarity desc, exclude already chosen
        filtered = [
            (i, sim.item()) for i, sim in enumerate(sims) if i not in chosen_set
        ]
        filtered_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)
        # Convert to 1-based IDs
        preds = [
            {**heroes_lookup[i + 1], "score": score}
            for i, score in filtered_sorted
            if i + 1 != 24
        ]

        return {"next_slot": n + 1, "sorted_candidates": preds}

    # Predict 5th hero using model when n == 4
    hero_tensor = torch.tensor([heroes_list], dtype=torch.long)
    role_tensor = torch.tensor([[role_map[h] for h in heroes_list]], dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(hero_tensor, role_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    k = probs.size(0)
    top_probs, top_indices = torch.topk(probs, k)
    preds = [
        # Restore to 1-based indexing
        {**heroes_lookup[int(idx) + 1], "score": float(prob)}
        for idx, prob in zip(top_indices, top_probs)
        if idx + 1 != 24
    ]

    return {"next_slot": 5, "sorted_candidates": preds}
