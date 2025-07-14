import asyncio
import httpx
import json

async def test_conversation_styles():
    """Test different conversation styles with the same input."""
    base_url = "http://localhost:8000"
    
    print(" Testing Different Conversation Styles\n")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # First, get available styles
        try:
            response = await client.get(f"{base_url}/conversation-styles")
            if response.status_code == 200:
                styles_info = response.json()
                print(" Available Styles:")
                for style, desc in styles_info["available_styles"].items():
                    print(f"  • {style}: {desc}")
                print(f"  • Default: {styles_info['default_style']}")
                print(f"  • Brand: {styles_info['current_brand']}\n")
            else:
                print(" Could not fetch available styles")
                return
        except Exception as e:
            print(f" Error connecting to server: {e}")
            print("Make sure the server is running with 'python server_test.py'")
            return
        
        # Test each style with preamble (greeting)
        styles = ["default", "casual", "genz"]
        
        for style in styles:
            print(f" Testing Style: {style.upper()}")
            print("-" * 50)
            
            # Test preamble with different styles
            preamble_data = {
                "chat_history": [],
                "config": {
                    "conversation_style": style,
                    "deployment": None
                }
            }
            
            try:
                response = await client.post(
                    f"{base_url}/preamble",
                    json=preamble_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    content = ""
                    async for chunk in response.aiter_text():
                        content += chunk
                    print(f"Greeting: {content}")
                else:
                    print(f" Error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f" Failed: {e}")
            
            print("\n")

if __name__ == "__main__":
    asyncio.run(test_conversation_styles())