from contextlib import asynccontextmanager
import re
from fastapi import FastAPI, Request, HTTPException
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from config import line_bot_api, handler
from services.process_job_link import process_job_url
from services.predict import FraudPredictor
from services.ui_renderer import create_risk_flex_message

from services.ui_renderer import create_risk_flex_message
from utils import download_multiple
from typing import cast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Job Scam Detector")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 下載模型 & scaler
    BUCKET = "muoshuei-bucket"
    FILES = ["model/fraud_detection_model.pth", "model/scaler.pkl"]
    model_paths = download_multiple(BUCKET, FILES)

    # 初始化 FraudPredictor
    app.state.predictor = FraudPredictor(model_path="tmp/model/fraud_detection_model.pth", scaler_path="tmp/model/scaler.pkl")
    logger.info("Init predictor")
    yield  # app 進入運行階段
    
    # shutdown 可釋放資源
    del app.state.predictor
    print("Predictor resources cleaned up.")
    logger.info("Terminated")

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
        logger.info(f"Searching job info for target_url: {target_url}")
        job_data = process_job_url(target_url)
        
        #TODO threads

        if not job_data:
             reply = TextSendMessage(text="❌ 無法讀取職缺資料，請確認該職缺是否已下架。")
             line_bot_api.reply_message(event.reply_token, reply)
             return
        logger.info(f"Got job data for {target_url}")
        # risk_result = analyze_risk(job_data)
        predictor = cast(FraudPredictor, app.state.predictor)

        sample_data = {
            'full_content': '急徵在家工作人員，日領薪水，加LINE ID: scam123',
            'salary_min': 50000,
            'salary_max': 100000,
            'salary_type': 1,
            'capital_amount_cleaned': 0,
            'employees_cleaned': 0,
            '供需人數 (應徵人數) (Number of Applicants)': 10,
            '縣市 (City/County)': 1,
            '工作經歷 (Work Experience)': 0,
            '學歷要求 (Educational Requirements)': 0
        }
        risk_result = predictor.predict(sample_data)

        flex_message = create_risk_flex_message(risk_result)

        line_bot_api.reply_message(event.reply_token, flex_message)

    else:
        # 選項 B: 引導使用者 (適合一對一)
        reply_text = "請貼上 104 職缺連結 (例如: https://www.104.com.tw/job/xxxxx)，我會幫您分析風險。"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)