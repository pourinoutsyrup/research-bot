import requests
async def send_to_discord(webhook_url: str, message: str):
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json={"content": message})
    except:
        pass