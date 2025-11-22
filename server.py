import re
from fastapi import FastAPI, Request, HTTPException
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from config import line_bot_api, handler
from services.process_job_link import process_job_url
from services.predict import FraudPredictor
from services.ui_renderer import create_risk_flex_message
from services.gemini_explain_risk import get_job_fraud_analysis

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
        job_data = process_job_url(target_url)
        
        if job_data.empty:
             reply = TextSendMessage(text="❌ 無法讀取職缺資料，請確認該職缺是否已下架。")
             line_bot_api.reply_message(event.reply_token, reply)
             return

        gemini_job_fraud_caption = get_job_fraud_analysis(job_data.head(1)) 
        print(gemini_job_fraud_caption)

        predictor = FraudPredictor()
        predict_risk = predictor.predict_csv(job_data)
        reply_string = TextSendMessage(text="predict_risk")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_string))
        print("Prediction Result:")
        print(predict_risk)
        
        # flex_message = create_risk_flex_message(predict_risk) # 還要重新設計
        
        # line_bot_api.reply_message(event.reply_token, flex_message)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=gemini_job_fraud_caption))

    else:
        # 選項 B: 引導使用者 (適合一對一)
        reply_text = "請貼上 104 職缺連結 (例如: https://www.104.com.tw/job/xxxxx)，我會幫您分析風險。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)