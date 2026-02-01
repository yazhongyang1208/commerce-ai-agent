import json
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB (Local persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection to remove Chinese data
try:
    chroma_client.delete_collection("products")
    print("  Old data deleted.")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(name="products")

# Load English data
with open("products.json", "r", encoding="utf-8") as f:
    products = json.load(f)

print(f" Processing {len(products)} products...")

# Ingest data
for product in products:
    # Combine text for better semantic search
    text = f"{product['name']} {product['description']} {product['category']}"
    
    # Store original JSON string in metadata for easy retrieval
    product_json_str = json.dumps(product, ensure_ascii=False)
    
    collection.add(
        ids=[product['id']],
        documents=[text],
        metadatas=[{"original_json": product_json_str}]
    )

print(" Data ingestion complete!")