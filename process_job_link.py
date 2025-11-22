import sys
import requests
from urllib.parse import urlparse
import pandas as pd
import re
import os
import json
import argparse

# ==============================
# Part 1: Crawler Logic (from 104/104_from_link.py)
# ==============================

BASE_JOB_URL = "https://www.104.com.tw/job/ajax/content/{job_id}"

COMMON_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

def extract_job_id_from_url(job_url: str) -> str | None:
    if job_url.startswith("//"):
        job_url = "https:" + job_url
    try:
        parsed = urlparse(job_url)
        path = parsed.path
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0].lower() == "job":
            return parts[1]
        if len(parts) == 1:
            return parts[0]
    except Exception:
        return None
    return None

def fetch_job_detail(job_id: str) -> dict | None:
    url = BASE_JOB_URL.format(job_id=job_id)
    headers = COMMON_HEADERS.copy()
    headers["Referer"] = f"https://www.104.com.tw/job/{job_id}"
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != requests.codes.ok:
            print(f"[Error] Job detail request failed for job_id={job_id}: {resp.status_code}")
            return None
        data = resp.json()
        return data.get("data", {})
    except Exception as e:
        print(f"[Error] Exception fetching job detail: {e}")
        return None

def join_list_field(field_val):
    if isinstance(field_val, list):
        if field_val and isinstance(field_val[0], dict):
            return "、".join(x.get("description", "") for x in field_val if x)
        return "、".join(str(x) for x in field_val)
    return field_val

def parse_job_detail_to_record(detail: dict, job_id: str | None) -> dict:
    header = detail.get("header", {}) if detail else {}
    job_detail = detail.get("jobDetail", {}) if detail else {}
    condition = detail.get("condition", {}) if detail else {}
    company = detail.get("company", {}) if detail else {}

    categories = job_detail.get("jobCategory", []) or []
    main_cat = categories[0]["description"] if categories else None
    sub_cat = categories[1]["description"] if len(categories) > 1 else None

    tools = join_list_field(condition.get("specialty"))
    skills = join_list_field(condition.get("skill"))

    company_tags = []
    if company.get("industryDesc"):
        company_tags.append(company["industryDesc"])
    if isinstance(company.get("tags"), list):
        company_tags.extend([str(t) for t in company["tags"]])
    company_tags_str = "、".join(company_tags) if company_tags else None

    city = job_detail.get("addressRegion") or ""
    area = job_detail.get("addressArea") or ""
    addr = job_detail.get("address") or ""
    full_address = f"{city}{area}{addr}".strip()

    record = {
        "職缺類別 (Job Category)": main_cat,
        "職位類別 (Position Category)": sub_cat,
        "職位 (Position)": header.get("jobName"),
        "縣市 (City/County)": job_detail.get("addressRegion"),
        "地區 (District/Area)": job_detail.get("addressArea"),
        "供需人數 (應徵人數) (Number of Applicants)": 0, # Default to 0 as API doesn't provide it easily
        "公司名稱 (Company Name)": header.get("custName") or company.get("custName"),
        "職缺名稱 (Job Title)": header.get("jobName"),
        "工作內容 (Job Description)": job_detail.get("jobDescription"),
        "職務類別 (Job Type)": job_detail.get("jobTypeDesc") or job_detail.get("jobType"),
        "工作待遇 (Salary)": job_detail.get("salary"),
        "工作性質 (Nature of Work)": job_detail.get("jobTypeDesc") or job_detail.get("jobType"),
        "上班地點 (Work Location)": full_address,
        "管理責任 (Management Responsibility)": job_detail.get("manageResp"),
        "上班時段 (Working Hours)": job_detail.get("workPeriod") or job_detail.get("workTime"),
        "需求人數 (Number of Positions)": job_detail.get("needEmp"),
        "工作經歷 (Work Experience)": condition.get("workExp"),
        "學歷要求 (Educational Requirements)": condition.get("edu"),
        "科系要求 (Departmental Requirements)": join_list_field(condition.get("major")),
        "擅長工具 (Tools Proficiency)": tools,
        "工作技能 (Job Skills)": skills,
        "其他條件 (Other Conditions)": condition.get("other"),
        "資本額 (Capital Amount)": company.get("capital"),
        "員工人數 (Number of Employees)": company.get("employee"),
        "公司標籤 (Company Tags)": company_tags_str,
    }
    
    if job_id:
        record["job_id"] = job_id
        record["104_url"] = f"https://www.104.com.tw/job/{job_id}"

    return record

# ==============================
# Part 2: Data Cleaning Logic (from data/merge_datasets.py)
# ==============================

def parse_salary(salary_str):
    if not isinstance(salary_str, str):
        return 0, 0, 0
    salary_str = salary_str.replace(',', '').strip()
    salary_type = 0
    if '月薪' in salary_str: salary_type = 1
    elif '日薪' in salary_str: salary_type = 2
    elif '時薪' in salary_str: salary_type = 3
    elif '年薪' in salary_str: salary_type = 4
    elif '面議' in salary_str: return 0, 0, 0
        
    numbers = re.findall(r'\d+', salary_str)
    if not numbers:
        return 0, 0, salary_type
    numbers = [int(n) for n in numbers]
    if len(numbers) == 1:
        min_sal = numbers[0]
        max_sal = numbers[0]
    else:
        min_sal = min(numbers)
        max_sal = max(numbers)
    return min_sal, max_sal, salary_type

def parse_capital(capital_str):
    if not isinstance(capital_str, str):
        return 0
    capital_str = capital_str.strip()
    if '暫不提供' in capital_str or not capital_str:
        return 0
    val = 0
    try:
        clean_str = capital_str.replace(',', '')
        num_match = re.search(r'[\d\.]+', clean_str)
        if not num_match:
            return 0
        num_val = float(num_match.group())
        if '億' in clean_str:
            val = num_val * 100000000
            if '萬' in clean_str:
                parts = clean_str.split('億')
                if len(parts) > 1 and '萬' in parts[1]:
                    wan_part = re.search(r'[\d\.]+', parts[1])
                    if wan_part:
                        val += float(wan_part.group()) * 10000
        elif '萬' in clean_str:
            val = num_val * 10000
        else:
            val = num_val
    except Exception:
        return 0
    return int(val)

def clean_employees(emp_str):
    if pd.isna(emp_str):
        return 0
    if isinstance(emp_str, (int, float)):
        return int(emp_str)
    if isinstance(emp_str, str):
        emp_str = emp_str.replace(',', '')
        nums = re.findall(r'\d+', emp_str)
        if nums:
            return int(nums[0])
    return 0

# ==============================
# Part 3: Feature Engineering Logic (from data/select_features.py)
# ==============================

def extract_simple_text_features(df):
    # 1. 描述長度
    df['desc_length'] = df['full_content'].astype(str).apply(len)
    
    # 2. 是否包含 LINE 關鍵字
    line_pattern = r'(?i)line\s*(?:id|:|：|\s)|加\s*line'
    df['has_line_keyword'] = df['full_content'].astype(str).apply(lambda x: 1 if re.search(line_pattern, x) else 0)
    
    # 3. 是否包含 "在家工作" / "輕鬆" 等關鍵字
    wfh_pattern = r'在家工作|居家辦公|輕鬆|簡單|免經驗|日領|現領'
    df['has_high_risk_keywords'] = df['full_content'].astype(str).apply(lambda x: 1 if re.search(wfh_pattern, x) else 0)
    
    return df

# ==============================
# Part 4: Mapping Logic
# ==============================

def get_mappings_from_dataset():
    """
    Generate mappings directly from the dataset in memory.
    Does not read or write any JSON files.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, 'merged_jobs_dataset.csv')
    
    print(f"Generating category mappings from {dataset_path}...")
    
    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset file {dataset_path} not found. Cannot generate mappings.")
        return None
        
    try:
        df = pd.read_csv(dataset_path)
        categorical_features = [
            '縣市 (City/County)', 
            '工作經歷 (Work Experience)', 
            '學歷要求 (Educational Requirements)'
        ]
        
        mappings = {}
        for col in categorical_features:
            # Fill NaNs as 'Unknown'
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                unique_vals = sorted(df[col].astype(str).unique())
                col_mapping = {val: idx for idx, val in enumerate(unique_vals)}
                mappings[col] = col_mapping
                print(f"Generated mapping for {col}: {len(col_mapping)} categories")
            else:
                print(f"[Warning] Column {col} not found in dataset.")
                mappings[col] = {}

        return mappings
        
    except Exception as e:
        print(f"[Error] Failed to generate mappings: {e}")
        return None

# ==============================
# Main Pipeline
# ==============================

def process_job_url(job_url, output_path='processed_job.csv'):
    print(f"Processing URL: {job_url}")
    
    # 1. Crawl
    job_id = extract_job_id_from_url(job_url)
    if not job_id:
        print(f"[Error] Cannot extract job_id from URL: {job_url}")
        return
    
    print(f"Job ID: {job_id}")
    detail = fetch_job_detail(job_id)
    if not detail:
        print("[Error] Failed to fetch job detail")
        return
        
    record = parse_job_detail_to_record(detail, job_id)
    
    # Convert to DataFrame (1 row)
    df = pd.DataFrame([record])
    
    # 2. Preprocessing (Merge Logic)
    print("Preprocessing data...")
    
    # Salary
    salary_features = df['工作待遇 (Salary)'].apply(parse_salary)
    df['salary_min'] = [x[0] for x in salary_features]
    df['salary_max'] = [x[1] for x in salary_features]
    df['salary_type'] = [x[2] for x in salary_features]
    
    # Capital
    df['資本額 (Capital Amount)'] = df['資本額 (Capital Amount)'].astype(str)
    df['capital_amount_cleaned'] = df['資本額 (Capital Amount)'].apply(parse_capital)
    
    # Employees
    df['employees_cleaned'] = df['員工人數 (Number of Employees)'].apply(clean_employees)
    
    # Text Combination
    text_cols = ['職缺名稱 (Job Title)', '工作內容 (Job Description)', '其他條件 (Other Conditions)', '公司名稱 (Company Name)']
    for col in text_cols:
        df[col] = df[col].fillna('')
        
    df['full_content'] = (
        df['職缺名稱 (Job Title)'] + ' ' + 
        df['工作內容 (Job Description)'] + ' ' + 
        df['其他條件 (Other Conditions)'] + ' ' +
        df['公司名稱 (Company Name)']
    )
    
    # Fill NaNs for categorical
    cat_cols = ['職缺類別 (Job Category)', '縣市 (City/County)', '地區 (District/Area)', '上班地點 (Work Location)', '工作經歷 (Work Experience)', '學歷要求 (Educational Requirements)']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
        
    # 3. Feature Selection & Engineering (Select Features Logic)
    print("Extracting features...")
    
    # Numeric features
    numeric_features = [
        'salary_min', 
        'salary_max', 
        'salary_type', 
        'capital_amount_cleaned', 
        'employees_cleaned',
        '供需人數 (應徵人數) (Number of Applicants)'
    ]
    
    # Categorical features
    categorical_features = [
        '縣市 (City/County)', 
        '工作經歷 (Work Experience)', 
        '學歷要求 (Educational Requirements)'
    ]
    
    # Load and apply mappings
    mappings = get_mappings_from_dataset()
    
    if mappings:

        with open("mapping.json", "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=4)
        for col in categorical_features:
            if col in mappings:
                col_map = mappings[col]
                # Ensure the column is string before mapping
                df[col] = df[col].astype(str).map(col_map)
                # Fill missing values (new categories) with -1 or 'Unknown' code
                unknown_code = col_map.get('Unknown', -1)
                df[col] = df[col].fillna(unknown_code).astype(int)
            else:
                print(f"[Warning] No mapping found for column: {col}")
    else:
        print("[Warning] Proceeding without categorical encoding (mappings missing).")
    
    # Text features
    text_features = ['full_content']
    
    # Target (Default to 0 for real 104 jobs)
    df['fraud_label'] = 0
    target = ['fraud_label']
    
    # Extract engineered text features
    df = extract_simple_text_features(df)
    added_text_features = ['desc_length', 'has_line_keyword', 'has_high_risk_keywords']
    
    # Final Column Selection
    selected_columns = numeric_features + categorical_features + text_features + added_text_features + target
    
    # Ensure all columns exist
    for col in selected_columns:
        if col not in df.columns:
            df[col] = 0 # or appropriate default
            
    df_final = df[selected_columns]
    
    print("-" * 30)
    print("Processed Record:")
    print(df_final.iloc[0])
    print("-" * 30)
    
    print(f"Saving to {output_path}...")
    df_final.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process 104 Job Link and convert to model-ready format.")
    parser.add_argument("url", help="The 104 Job URL (e.g., https://www.104.com.tw/job/8v3a2)")
    parser.add_argument("--output", default="processed_job1.csv", help="Output CSV path (default: processed_job.csv)")
    
    args = parser.parse_args()
    process_job_url(args.url, args.output)
