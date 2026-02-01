AI Commerce Agent

A full-stack AI-powered shopping assistant capable of natural conversation, text-based product search, and image-based product discovery.

Built with **FastAPI**, **Zhipu AI (GLM-4)**, **ChromaDB**, and **Vanilla JS**.

Features

-   **Conversational AI**: Handles general chitchat and shopping inquiries distinctively using **Intent Recognition**.
-   **RAG (Retrieval-Augmented Generation)**: Recommends products based on semantic similarity from a vector database.
-   **Visual Search**: Allows users to upload images to find similar products using Multimodal AI (Vision).
-   **Smart Filtering**: Uses distance thresholds to ensure only relevant products are recommended (no hallucinations).
-   **Lightweight Frontend**: Single-file HTML/JS interface with no build step required.

Tech Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Backend** | Python (FastAPI) | High performance, easy async support, and native Swagger documentation. |
| **LLM & Vision** | Zhipu AI (GLM-4) | Cost-effective, high availability, and strong multimodal capabilities. |
| **Vector DB** | ChromaDB | Serverless, local persistence, and easy integration for Python. |
| **Frontend** | HTML5 + Tailwind CSS | Zero-dependency, fast prototyping, and responsive design. |

Project Structure

├── main.py           # The backend API server (FastAPI)
├── ingest_data.py    # Script to load product data into Vector DB
├── products.json     # The mock product catalog (Data source)
├── index.html        # The user interface (Frontend)
├── chroma_db/        # Persisted vector database (Generated after ingestion)
└── README.md         # Project documentation

Quick Start:
1. Prerequisites
    Python 3.8+ installed.
    An API Key from Zhipu AI.

2. Installation
    Clone the repository (or download the files) and install the dependencies:
        pip install fastapi uvicorn chromadb zhipuai python-multipart

3. Configuration
    Open main.py and replace the placeholder with your actual API Key:
        ZHIPU_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"

4. Data Ingestion (Build the Brain)
    Before running the server, you must load the product data into the vector database. Run this script once:
        python ingest_data.py

5. Run the Server
    Start the backend server:
        python main.py

6. Launch the App
    Simply double-click index.html to open it in your browser. Alternatively, serve it using a simple HTTP server:
        python -m http.server 3000