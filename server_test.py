"""
Simple server launcher for the Ordering ChatBot API
"""
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
def get_required_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. Please set it in your .env file."
        )
    return value

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Get brand name and default conversation style from environment
BRAND_NAME = os.getenv("BRAND_NAME")
CONVERSATION_STYLE = os.getenv("CONVERSATION_STYLE", "default")  # Make it optional with default

def main():
    """Launch the FastAPI server."""
    # Validate required environment variables
    required_var_names = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "RESTAURANT_BRAND",
        "CONVERSATION_STYLE"
    ]
    
    missing_vars = []
    for var_name in required_var_names:
        try:
            get_required_env_var(var_name)
        except ValueError:
            missing_vars.append(var_name)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file.")
        return
    
    print(f" Starting Ordering ChatBot API server...")
    print(f" Brand: {os.getenv('RESTAURANT_BRAND')}")
    print(f" Default Style: {os.getenv('CONVERSATION_STYLE')}")
    print(f" Access at: http://localhost:8000")
    print(f" API Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "streaming_ordering_chatbot.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()