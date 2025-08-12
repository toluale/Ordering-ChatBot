import asyncio
import time
import logging
from typing import Optional, Dict, Any
from uuid import uuid4
import httpx
import streamlit as st

logger = logging.getLogger(__name__)

class HTTPConnectionManager:
    """Simple HTTP connection manager that creates fresh clients for each request to avoid Streamlit event loop issues."""
    
    def __init__(self):
        pass  # No persistent state to avoid event loop issues
    
    def _create_client(self) -> httpx.AsyncClient:
        """Create a new HTTP client optimized for single-use with Streamlit."""
        # Simple timeout settings without connection pooling
        timeout = httpx.Timeout(
            connect=15.0,     # Connection timeout
            read=120.0,       # Long read timeout for streaming
            write=30.0,       # Write timeout
        )
        
        return httpx.AsyncClient(
            timeout=timeout,
            http2=False,      # Disable HTTP/2 for simplicity
            follow_redirects=True
        )
    
    async def get_client(self, session_id: Optional[str] = None) -> httpx.AsyncClient:
        """Create a fresh HTTP client for each request to avoid event loop issues."""
        return self._create_client()
    
    async def close_session_client(self, session_id: str):
        """No-op since we don't maintain persistent clients."""
        pass
    
    async def close(self):
        """No-op since we don't maintain persistent clients."""
        pass
    
    async def request(
        self, 
        method: str, 
        url: str, 
        session_id: Optional[str] = None,
        **kwargs
    ) -> httpx.Response:
        """Make an HTTP request with fallback error handling."""
        max_retries = 2
        last_error = None
        
        for attempt in range(max_retries):
            client = None
            try:
                # Get session ID from Streamlit if not provided and available
                try:
                    import streamlit as st
                    actual_session_id = session_id or getattr(st, 'session_state', {}).get('session_id', None)
                except (ImportError, AttributeError):
                    actual_session_id = session_id
                
                client = await self.get_client(actual_session_id)
                
                # Add default headers
                headers = kwargs.get('headers', {})
                headers.update({
                    "brand-session-id": actual_session_id or str(uuid4()),
                    "request-id": str(uuid4()),
                })
                kwargs['headers'] = headers
                
                response = await client.request(method, url, **kwargs)
                return response
                
            except (RuntimeError, httpx.RequestError) as e:
                last_error = e
                error_str = str(e)
                
                if "Event loop is closed" in error_str or "Connection pool is closed" in error_str:
                    logger.warning(f"Connection issue on attempt {attempt + 1}: {e}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt, create simple client
                        try:
                            simple_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
                            response = await simple_client.request(method, url, **kwargs)
                            await simple_client.aclose()
                            return response
                        except Exception as final_e:
                            raise httpx.RequestError(f"All connection attempts failed: {final_e}")
                else:
                    raise e
            finally:
                # Always close the client after use to prevent event loop issues
                if client and not client.is_closed:
                    try:
                        await client.aclose()
                    except:
                        pass
        
        raise httpx.RequestError(f"Request failed after {max_retries} attempts: {last_error}")
    
    async def stream_request(
        self, 
        method: str, 
        url: str, 
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Make a streaming HTTP request with fallback error handling."""
        max_retries = 2
        
        for attempt in range(max_retries):
            client = None
            try:
                # Get session ID from Streamlit if not provided and available
                try:
                    import streamlit as st
                    actual_session_id = session_id or getattr(st, 'session_state', {}).get('session_id', None)
                except (ImportError, AttributeError):
                    actual_session_id = session_id
                
                client = await self.get_client(actual_session_id)
                
                # Add default headers
                headers = kwargs.get('headers', {})
                headers.update({
                    "brand-session-id": actual_session_id or str(uuid4()),
                    "request-id": str(uuid4()),
                })
                kwargs['headers'] = headers
                
                async with client.stream(method, url, **kwargs) as response:
                    yield response
                return  # Successful completion
                
            except (RuntimeError, httpx.RequestError) as e:
                error_str = str(e)
                
                if "Event loop is closed" in error_str or "Connection pool is closed" in error_str:
                    logger.warning(f"Streaming connection issue on attempt {attempt + 1}: {e}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt, create simple client
                        try:
                            simple_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
                            async with simple_client.stream(method, url, **kwargs) as response:
                                yield response
                            await simple_client.aclose()
                            return
                        except Exception as final_e:
                            raise httpx.RequestError(f"All streaming attempts failed: {final_e}")
                else:
                    raise e
            finally:
                # Always close the client after use to prevent event loop issues
                if client and not client.is_closed:
                    try:
                        await client.aclose()
                    except:
                        pass

    async def cleanup(self):
        """Cleanup method - no-op since we use fresh clients for each request."""
        pass


# Global instance
http_manager = HTTPConnectionManager()

# Simplified cleanup for the new approach
def cleanup_session():
    """Cleanup HTTP client resources."""
    try:
        # Call the cleanup method instead of accessing private attributes
        asyncio.run(http_manager.cleanup())
    except:
        # If async cleanup fails, just pass
        pass

# Register cleanup (simplified approach)
try:
    import streamlit as st
    import atexit
    
    # Only register once per module import
    if not hasattr(cleanup_session, '_registered'):
        atexit.register(cleanup_session)
        cleanup_session._registered = True
        
except ImportError:
    # Streamlit not available
    pass