"""
Simple test client to verify the API endpoints work
"""
import asyncio
import json
import httpx
from typing import Dict, Any

async def test_api_endpoints():
    """Test the main API endpoints."""
    base_url = "http://localhost:8000"
    
    print("Testing API Endpoints...")
    print("Make sure the server is running with 'python server.py'\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Health check
        try:
            response = await client.get(f"{base_url}/conversation-styles")
            if response.status_code == 200:
                styles = response.json()
                print("GET /conversation-styles")
                print(f"   Available styles: {list(styles['available_styles'].keys())}")
            else:
                print(f"GET /conversation-styles failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Server connection failed: {e}")
            print("   Make sure server is running on http://localhost:8000")
            return False
        
        # Test 2: Screen endpoint
        print("\nTesting message screening...")
        screen_data = {
            "message": "I want a burger",
            "chat_history": [],
            "current_order": {"items": []}
        }
        
        try:
            response = await client.post(
                f"{base_url}/screen",
                json=screen_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                result = response.json()
                print("POST /screen")
                print(f"   Intent: {result.get('intent', 'unknown')}")
                print(f"   Message: {result.get('redacted_message', '')[:50]}...")
            else:
                print(f"POST /screen failed: {response.status_code}")
                print(f"   Response: {response.text}")
        except Exception as e:
            print(f"Screen test failed: {e}")
        
        # Test 3: Preamble endpoint
        print("\nTesting preamble generation...")
        preamble_data = {
            "chat_history": [],
            "config": {"conversation_style": "default", "deployment": None}
        }
        
        try:
            response = await client.post(
                f"{base_url}/preamble",
                json=preamble_data,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                print("POST /preamble")
                # Read streaming response
                content = ""
                async for chunk in response.aiter_text():
                    content += chunk
                print(f"   Response preview: {content[:100]}...")
            else:
                print(f"POST /preamble failed: {response.status_code}")
        except Exception as e:
            print(f"Preamble test failed: {e}")
        
        print("\nAPI tests completed!")
        return True

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())