import duckdb
import torch
import math
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

def extract_features():
    print("Connecting to DuckDB...")
    con = duckdb.connect("data/preprocessing.duckdb", read_only=True)

    # Fetch all markets sorted strictly by market_id so the tensor indices are fully deterministic
    # We coalesce nulls so the LLM doesn't crash on empty text
    df_markets = con.execute("""
        SELECT 
            market_id,
            COALESCE(question, '') AS question,
            COALESCE(description, '') AS description,
            COALESCE(tags, '') AS tags,
            deploying_timestamp
        FROM universe.markets
        ORDER BY market_id ASC
    """).df()

    num_markets = len(df_markets)
    print(f"Loaded {num_markets} markets.")

    # Save the deterministic integer mapping (Node ID -> String ID) so the GNN engine can correctly align edges!
    mapping_df = df_markets[['market_id']].reset_index().rename(columns={'index': 'node_index'})
    mapping_df.to_csv('data/derived/market_node_mapping.csv', index=False)
    print("Saved Node Alignment Dictionary to data/derived/market_node_mapping.csv")

    # 1. TEXT ENCODING (LLM)
    # Using a fast, high-quality semantic model (dim=384)
    model_name = 'all-MiniLM-L6-v2'
    print(f"Loading SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Forcing CPU to bypass the PyTorch HIP/ROCm driver crash
    device = 'cpu'

    print("Constructing semantic strings...")
    df_markets['semantic_text'] = (
        "Question: " + df_markets['question'] + 
        " | Description: " + df_markets['description'] + 
        " | Tags: " + df_markets['tags']
    )

    print(f"Encoding text strings via LLM on {device.upper()}... (This may take a few minutes)")
    t0 = time.time()
    # Encode in batches. show_progress_bar=True provides a nice tqdm terminal visual.
    text_embeddings = model.encode(
        df_markets['semantic_text'].tolist(), 
        batch_size=256, 
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )
    print(f"Text encoding completed in {time.time()-t0:.2f}s")
    
    # Move back to CPU for saving
    text_embeddings = text_embeddings.cpu()

    # 2. TEMPORAL ENCODING
    print("Generating Trigonometric Temporal Encodings (Time2Vec style)...")
    timestamps = pd.to_datetime(df_markets['deploying_timestamp'])

    # If any timestamp is null (e.g. broken API data), fill with epoch 0
    timestamps = timestamps.fillna(pd.Timestamp("2000-01-01"))

    hour = torch.tensor(timestamps.dt.hour.values, dtype=torch.float32)
    day_of_week = torch.tensor(timestamps.dt.dayofweek.values, dtype=torch.float32)
    month = torch.tensor(timestamps.dt.month.values, dtype=torch.float32)

    # Sine/Cosine encoding mathematically captures the cyclical nature of time
    # Hour: cycle of 24
    hour_sin = torch.sin(2 * math.pi * hour / 24.0)
    hour_cos = torch.cos(2 * math.pi * hour / 24.0)

    # Day of Week: cycle of 7
    dow_sin = torch.sin(2 * math.pi * day_of_week / 7.0)
    dow_cos = torch.cos(2 * math.pi * day_of_week / 7.0)

    # Month: cycle of 12
    month_sin = torch.sin(2 * math.pi * month / 12.0)
    month_cos = torch.cos(2 * math.pi * month / 12.0)

    temporal_embeddings = torch.stack([
        hour_sin, hour_cos, 
        dow_sin, dow_cos, 
        month_sin, month_cos
    ], dim=1)

    # 3. CONCATENATE AND SAVE
    print("Concatenating Final Feature Tensor...")
    final_market_features = torch.cat([text_embeddings, temporal_embeddings], dim=1)
    
    print(f"Final Tensor Shape: {final_market_features.shape}")
    # We expect [num_markets, 384 (text) + 6 (time)] = [175k, 390]

    out_path = "data/derived/market_features.pt"
    torch.save(final_market_features, out_path)
    print(f"Successfully saved frozen Market Initial Features to {out_path}!")

if __name__ == "__main__":
    extract_features()
