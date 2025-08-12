#!/usr/bin/env python3
"""
Test script to verify the simplified HTTP manager resolves event loop issues.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add the streaming_ordering_chatbot to the path
sys.path.insert(0, str(Path(__file__).parent / "streaming_ordering_chatbot" / "streamlit"))

from http_manager import http_manager

async def test_http_manager():
    """Test the simplified HTTP manager functionality."""
    print("Testing simplified HTTP manager...")
    
    try:
        # Test basic request
        print("1. Testing basic request...")
        response = await http_manager.request(
            "GET", 
            "https://httpbin.org/get",
            headers={"Test": "simplified-manager"}
        )
        print(f"   ✓ Basic request successful: {response.status_code}")
        
        # Test multiple requests to verify no connection reuse issues
        print("2. Testing multiple sequential requests...")
        for i in range(3):
            response = await http_manager.request(
                "GET", 
                f"https://httpbin.org/get?request={i+1}"
            )
            print(f"   ✓ Request {i+1} successful: {response.status_code}")
            
        # Test streaming request
        print("3. Testing streaming request...")
        async for response in http_manager.stream_request(
            "GET",
            "https://httpbin.org/stream/3"
        ):
            print(f"   ✓ Streaming request successful: {response.status_code}")
            async for chunk in response.aiter_lines():
                if chunk:
                    line = chunk.decode()
                    print(f"   ✓ Received streaming chunk: {len(line)} chars")
            break
            
        print("\n🎉 All tests passed! The simplified HTTP manager is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("=== Simplified HTTP Manager Test ===\n")
    
    # Test event loop compatibility
    try:
        result = asyncio.run(test_http_manager())
        if result:
            print("\n✅ HTTP manager is ready for Streamlit integration!")
            return 0
        else:
            print("\n❌ HTTP manager test failed!")
            return 1
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            print(f"❌ Event loop error detected: {e}")
            return 1
        else:
            raise

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
