import re
from fastapi import FastAPI, Request, HTTPException
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from config import line_bot_api, handler
from services.scraper import get_104_job_data
from services.analyzer import analyze_risk
from services.ui_renderer import create_risk_flex_message

app = FastAPI(title="Job Scam Detector")

@app.get("/")
async def root():
    return {"message": "Job Scam Detector Running"}

@app.post("/webhook")
async def webhook(request: Request):
    body = await request.body()
    signature = request.headers.get("X-Line-Signature", "")
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text.strip()

    url_pattern = r'https?://www\.104\.com\.tw/job/[a-zA-Z0-9]+'
    
    match = re.search(url_pattern, user_text)

    if match:
        target_url = match.group(0)
        job_data = get_104_job_data(target_url)
        
        if not job_data:
             reply = TextSendMessage(text="❌ 無法讀取職缺資料，請確認該職缺是否已下架。")
             line_bot_api.reply_message(event.reply_token, reply)
             return

        risk_result = analyze_risk(job_data)
        
        flex_message = create_risk_flex_message(risk_result)
        
        line_bot_api.reply_message(event.reply_token, flex_message)

    else:
        # 選項 B: 引導使用者 (適合一對一)
        reply_text = "請貼上 104 職缺連結 (例如: https://www.104.com.tw/job/xxxxx)，我會幫您分析風險。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)