


import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from zhipuai import ZhipuAI
import base64

# ================= CONFIGURATION =================

# PASTE YOUR ZHIPU API KEY HERE
ZHIPU_API_KEY = "d9cc16b860844200be0daafde75ace7e.4Yw8yLwJ827m5fi7"

# Initialize Client
client = ZhipuAI(api_key=ZHIPU_API_KEY)

# Connect to Database (Local persistent storage)
# Ensure you have run ingest_data.py first!
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="products")

# ================= APP SETUP =================
app = FastAPI(title="AI Commerce Agent (Final English Version)")

# Enable CORS so frontend can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= DATA MODELS =================
class ChatRequest(BaseModel):
    message: str

class Product(BaseModel):
    id: str
    name: str
    price: float
    description: str
    image_url: str

class ChatResponse(BaseModel):
    response: str
    recommended_products: List[Product] = []

# ================= HELPER FUNCTIONS =================

def search_products(query_text: str, n_results=5):
    """
    Search DB with a Relevance Threshold.
    Filters out results that are not similar enough.
    """
    try:
        # Fetch 5 candidates, including their distance scores
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        products = []
        if results['documents']:
            # Iterate through results and filter by distance
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                
                # ================= THRESHOLD FILTER =================
                # Lower distance = higher similarity.
                # If distance is > 1.3, it's likely irrelevant. Skip it.
                if distance > 1.3:
                    continue 
                # ====================================================

                meta = results['metadatas'][0][i]
                if 'original_json' in meta:
                    product_data = json.loads(meta['original_json'])
                    products.append(Product(**product_data))
        
        # Only return the filtered list (might be less than 5, or empty)
        return products
    except Exception as e:
        print(f" Search Error: {e}")
        return []

def analyze_intent(user_msg: str) -> bool:
    """
    Determine if user wants to buy/search (True) or just chat (False).
    """
    try:
        # Use GLM-4-Flash for fast, cheap classification
        response = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[
                {"role": "system", "content": "You are an intent classifier. If the user input is about buying products, searching for items, checking prices, or asking for recommendations, reply 'YES'. If it is small talk, greetings, or unrelated topics, reply 'NO'. Only reply with YES or NO."},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1, # Low temp for stability
            max_tokens=10
        )
        result = response.choices[0].message.content.strip().upper()
        print(f" Intent Analysis: '{user_msg}' -> {result}") 
        return "YES" in result
    except Exception as e:
        print(f" Intent Error (Defaulting to Search): {e}")
        return True

# ================= API ENDPOINTS =================

@app.get("/")
def read_root():
    return {"status": "AI Agent is ready."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_msg = request.message
        
        # Step 1: Analyze Intent
        is_shopping = analyze_intent(user_msg)
        
        relevant_products = []
        system_prompt = ""

        # Step 2: Build Context based on intent
        if is_shopping:
            print(" User wants to shop. Searching database...")
            # Search with filtering enabled
            relevant_products = search_products(user_msg)
            
            if relevant_products:
                # Found good matches after filtering
                context_str = "\n".join([f"- {p.name}: ${p.price}, {p.description}" for p in relevant_products])
                system_prompt = f"""
                You are a helpful e-commerce shopping assistant.
                
                Here are the high-relevance items found in inventory:
                {context_str}
                
                User Request: "{user_msg}"
                
                Please recommend products from the list above. Briefly highlight why they match the user's request.
                """
            else:
                # Either database is empty OR all results were filtered out by threshold
                system_prompt = f"""
                You are a helpful e-commerce shopping assistant.
                User Request: "{user_msg}"
                
                We searched our inventory but unfortunately could not find any items matching this specific request.
                Politely inform the user that we don't have it in stock right now.
                """
        else:
            print(" User is chatting (no search performed)...")
            relevant_products = []
            system_prompt = f"""
            You are a friendly AI Shopkeeper.
            The user is engaging in small talk: "{user_msg}"
            
            Reply politely and strictly in English. Do not recommend products unless asked.
            """

        # Step 3: Generate Response using GLM-4-Flash
        response = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[
                {"role": "user", "content": system_prompt}
            ],
        )
        
        ai_reply = response.choices[0].message.content
        
        return ChatResponse(
            response=ai_reply,
            recommended_products=relevant_products
        )
        
    except Exception as e:
        print(f" Chat Error: {str(e)}")
        return ChatResponse(response=f"Sorry, something went wrong on the server: {str(e)}")

@app.post("/search-image", response_model=ChatResponse)
async def search_image_endpoint(file: UploadFile = File(...)):
    """
    Image Search Flow: Vision API -> Text Search (Filtered) -> Final Response
    """
    try:
        print(" Image received. Processing...")
        
        # 1. Convert image to Base64 (Required for Zhipu Vision)
        image_data = await file.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # 2. Call GLM-4V (Vision Model) to describe the image
        response = client.chat.completions.create(
            model="glm-4v-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Look at this product image. Describe its category, color, material, and key style features. Output only the highly descriptive keywords in English for search purposes."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        description = response.choices[0].message.content
        print(f" Vision identified: {description}")
        
        # 3. Search database using the description (Filtered)
        relevant_products = search_products(description)
        
        # 4. Construct response
        if relevant_products:
             final_response = f"I see this is a: {description}. \nHere are the similar items I found in stock:"
        else:
             final_response = f"I see this is a: {description}. \nUnfortunately, I couldn't find any similar items in our current stock."

        return ChatResponse(
            response=final_response,
            recommended_products=relevant_products
        )
        
    except Exception as e:
        print(f" Image Error: {str(e)}")
        return ChatResponse(response=f"Failed to process image: {str(e)}")

if __name__ == "__main__":
    print(" Server starting on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)