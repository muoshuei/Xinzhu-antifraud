# services/analyzer.py

def analyze_risk(job_data: dict):
    """
    分析職缺資料並回傳風險等級與理由
    """
    if not job_data:
        return {"level": "unknown", "score": 0, "reasons": ["無法讀取職缺資料"]}

    score = 0
    reasons = []
    
    header = job_data.get('header', {})
    job_detail = job_data.get('jobDetail', {})
    contact = job_data.get('contact', {})

    job_desc = job_detail.get('jobDescription', '')
    salary = job_detail.get('salary', '')
    contact_info = contact.get('hrContact', '')
    
    # --- 風險規則 ---
    risk_keywords = ['輕鬆', '免經驗', '高薪', '在家工作', '博弈', '轉帳', '只要手機']
    found_keywords = [kw for kw in risk_keywords if kw in job_desc]
    
    if found_keywords:
        score += 30
        reasons.append(f"包含高風險關鍵字: {', '.join(found_keywords)}")

    if 'line' in contact_info.lower() or '加賴' in contact_info:
        score += 40
        reasons.append("要求私下加 Line 聯絡")

    if "面議" in salary and "兼職" in header.get('jobName', ''):
        score += 20
        reasons.append("兼職工作卻薪資面議")

    # 判定等級
    if score >= 60:
        level = "danger"
        title = "高風險 (High Risk)"
        color = "#FF3333"
    elif score >= 30:
        level = "warning"
        title = "需謹慎 (Caution)"
        color = "#FFAA33"
    else:
        level = "safe"
        title = "低風險 (Safe)"
        color = "#33AA33"

    if not reasons:
        reasons.append("未偵測到明顯風險特徵")

    return {
        "level": level,
        "title": title,
        "color": color,
        "score": score,
        "reasons": reasons,
        "job_name": header.get('jobName', '未知職缺'),
        "company": header.get('custName', '未知公司')
    }