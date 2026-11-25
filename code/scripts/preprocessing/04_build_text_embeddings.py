"""
Script to extract market metadata (question + description) and generate text embeddings
using a local SentenceTransformer model.

Embeddings are saved to data/derived/market_embeddings.pt as a dictionary
mapping market_id -> torch.Tensor.
"""

import duckdb
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

def build_market_embeddings():
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    db_path = project_root / 'data' / 'preprocessing.duckdb'
    out_path = project_root / 'data' / 'derived' / 'market_embeddings.pt'
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading active markets for the test week...")
    db = duckdb.connect(str(db_path), read_only=True)
    df = db.execute("""
        WITH active_markets AS (
            SELECT market_id, SUM(volume_2s) as total_vol
            FROM read_parquet('data/derived/multi_day/transactions/*.parquet')
            GROUP BY market_id
            ORDER BY total_vol DESC
            LIMIT 500
        )
        SELECT m.market_id, m.question, m.description 
        FROM universe.markets m
        JOIN active_markets a ON m.market_id = a.market_id
    """).df()
    db.close()
    
    print(f"Loaded {len(df)} markets.")
    
    # Fill missing values and combine text
    df['question'] = df['question'].fillna("")
    df['description'] = df['description'].fillna("")
    
    # Construct input text: "Question: <q> \n Description: <d>"
    texts = []
    for _, row in df.iterrows():
        text = f"Question: {row['question']}\nDescription: {row['description']}"
        texts.append(text)
        
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    # This is a small (80MB) and fast model that produces 384-dim embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding texts into embeddings...")
    embeddings = model.encode(
        texts, 
        batch_size=128, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        device=torch.device('cpu')
    )
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create mapping
    emb_dict = {}
    for i, market_id in enumerate(df['market_id']):
        emb_dict[market_id] = embeddings[i].cpu()
        
    print(f"Saving to {out_path}...")
    torch.save(emb_dict, out_path)
    print("Done!")

if __name__ == "__main__":
    build_market_embeddings()
