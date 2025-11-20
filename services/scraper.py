# services/scraper.py
import re
import requests

def get_104_job_data(url: str):
    """
    從 104 URL 中提取 Job ID 並呼叫其隱藏的 AJAX API 獲取資料
    """
    match = re.search(r'job/([a-zA-Z0-9]+)', url)
    if not match:
        return None
    
    job_id = match.group(1)
    api_url = f"https://www.104.com.tw/job/ajax/content/{job_id}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Referer": url
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            return response.json().get('data')
        return None
    except Exception as e:
        print(f"Scraping Error: {e}")
        return None
    
if __name__ == "__main__":
    url = "https://www.104.com.tw/job/8msz8"

    import json
    data = get_104_job_data(url)
    json_str = json.dumps(data)
    with open("data.json", "w") as f:
        f.write(json_str)
        
