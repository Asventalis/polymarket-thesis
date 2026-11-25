import torch
from pathlib import Path

BASE_DIR = Path('/data/v.pampinella/agy_project')
market_init = torch.load(BASE_DIR / 'data/derived/market_features.pt', map_location='cpu')
print('Market init contains NaNs:', torch.isnan(market_init).any().item())
print('Market init contains Infs:', torch.isinf(market_init).any().item())
