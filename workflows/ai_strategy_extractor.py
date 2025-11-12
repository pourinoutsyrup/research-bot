"""
workflows/ai_strategy_extractor.py
ATHENA v3.0 — Real DeepSeek API + JSON Extraction
"""
import logging
import json
import httpx

logger = logging.getLogger(__name__)

class DeepSeekExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    async def extract_tradable_strategy(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        logger.info(f"Making API call with content length: {len(prompt)}")
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.base_url, json=payload, headers=headers)
            
            logger.info(f"API Response Status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"DeepSeek API error: {response.text}")
                return {"strategy_code": "[]"}

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Extract JSON from ```json ... ```
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                json_str = content.strip()

            try:
                parsed = json.loads(json_str)
                result = json.dumps(parsed)
            except:
                # Fallback: return raw text
                result = json_str

            return {"strategy_code": result}

        except Exception as e:
            logger.error(f"DeepSeek extraction failed: {e}")
            return {"strategy_code": "[]"}