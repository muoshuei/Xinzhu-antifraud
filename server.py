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

    # 路由邏輯
    if user_text.startswith("/scan"):
        parts = user_text.split(maxsplit=1)
        if len(parts) < 2:
            reply = TextSendMessage(text="請輸入正確格式：/scan [104職缺連結]")
            line_bot_api.reply_message(event.reply_token, reply)
            return

        url = parts[1]
        
        # 步驟 1: 獲取資料 (Scraper)
        job_data = get_104_job_data(url)
        
        if not job_data:
             reply = TextSendMessage(text="❌ 無法讀取職缺資料，請確認連結是否為 104 有效職缺。")
             line_bot_api.reply_message(event.reply_token, reply)
             return

        # 步驟 2: 分析風險 (Analyzer)
        risk_result = analyze_risk(job_data)
        
        # 步驟 3: 產生 UI (Renderer)
        flex_message = create_risk_flex_message(risk_result)
        
        # 步驟 4: 回覆 (Bot API)
        line_bot_api.reply_message(event.reply_token, flex_message)

    else:
        # 其他訊息處理
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)