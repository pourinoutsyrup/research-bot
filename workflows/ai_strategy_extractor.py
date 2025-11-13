import aiohttp
import json
import logging
import os
from typing import Dict, Any, Optional

load_dotenv()

class DeepSeekClient:
    """Real DeepSeek API client"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1"):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
        
        self.base_url = base_url
        self.logger = logging.getLogger("deepseek")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def extract_tradable_strategy(self, prompt: str) -> Dict[str, Any]:
        """Make real DeepSeek API call"""
        try:
            return await self._make_api_call(prompt)
        except Exception as e:
            self.logger.error(f"DeepSeek API call failed: {e}")
            return {"error": str(e)}
    
    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """Make actual API call to DeepSeek"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 4000,
            "stream": False
        }
        
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_response(data)
            else:
                error_text = await response.text()
                self.logger.error(f"API call failed: {response.status} - {error_text}")
                raise Exception(f"API call failed: {response.status} - {error_text}")
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse DeepSeek API response"""
        try:
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                
                # Try to parse as JSON first
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # If not JSON, return as text with some structure
                    return {
                        "content": content,
                        "raw_response": content
                    }
            else:
                return {"error": "No choices in response", "raw_response": response}
                
        except Exception as e:
            self.logger.error(f"Failed to parse DeepSeek response: {e}")
            return {"error": f"Parse error: {str(e)}", "raw_response": response}
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()