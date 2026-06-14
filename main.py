from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os, ast, json
from sklearn.linear_model import LinearRegression
import anthropic

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_dir = os.path.dirname(os.path.abspath(__file__))
MAX_NEW  = 100
WINDOW   = 2
ALL_YEARS = [65, 66, 67, 68]
REQUIRED_YEARS = {65, 66, 67, 68}
NAN_PLACEHOLDER = '__NAN__'

# =====================================================================
# ส่วนที่ 1: โหลดข้อมูลทุกอย่างตอนเปิด server
# =====================================================================

def safe_read_csv(path):
    for enc in ['utf-8', 'utf-8-sig', 'cp874', 'tis-620']:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)

def find_file(filename):
    """หาไฟล์ใน data/ หรือ root"""
    for base in [os.path.join(current_dir, 'data'), current_dir]:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return None

# ------------------------------------------------------------------
# 1A. โหลด Excel ปี 65-68 → สร้างข้อมูลสำหรับ ML และ Frontend
# ------------------------------------------------------------------
all_data = []
min_lookup_raw = {65: {}, 66: {}, 67: {}, 68: {}} # เก็บคะแนนดิบให้หน้าเว็บ Frontend

YEAR_CONFIG = {
    65: {
        'file':    'TCAS65_maxmin.xlsx',
        'id_col':  'program_id',
        'min_col': None,
        'det_col': 'project_name_th',
        'app_col': 'สมัคร',
        'acc_col': 'รับ',
        'fac_col': 'program_lookup_programs.faculty_name_th',
        'ins_col': 'university_name',
    },
    66: {
        'file':    'TCAS66_maxmin.xlsx',
        'id_col':  'รหัสหลักสูตร',
        'min_col': 'คะแนนต่ำสุด',
        'det_col': 'รายละเอียด',
        'app_col': 'สมัคร',
        'acc_col': 'รับ',
        'fac_col': 'คณะ/สำนักวิชา',
        'ins_col': 'สถาบัน',
    },
    67: {
        'file':    'TCAS67_maxmin.xlsx',
        'id_col':  'รหัสหลักสูตร',
        'min_col': None,
        'det_col': 'รายละเอียด',
        'app_col': 'สมัคร',
        'acc_col': 'รับ',
        'fac_col': 'คณะ',
        'ins_col': 'สถาบัน',
    },
    68: {
        'file':    'TCAS68_maxmin.xlsx',
        'id_col':  'รหัสหลักสูตร',
        'min_col': None,
        'det_col': 'รายละเอียด',
        'app_col': 'สมัคร',
        'acc_col': 'รับ',
        'fac_col': 'คณะ',
        'ins_col': 'สถาบัน',
    },
}

for year, cfg in YEAR_CONFIG.items():
    fpath = find_file(cfg['file'])
    if not fpath:
        print(f"⚠️ ไม่พบไฟล์ {cfg['file']}")
        continue

    df = pd.read_excel(fpath)
    df['year'] = year

    if year == 65:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด หลังประมวลผลรอบ 2'].fillna(df.get('คะแนนต่ำสุด'))
        df['รหัสหลักสูตร'] = df['program_id']
    elif year == 67:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด หลังประมวลผลรอบ 2'].fillna(df.get('คะแนนต่ำสุด'))
    elif year == 68:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด ประมวลผลครั้งที่ 2'].fillna(df.get('คะแนนต่ำสุด ประมวลผลครั้งที่ 1'))

    df['min_pct'] = df['คะแนนต่ำสุด'] / MAX_NEW * 100
    df['รหัสหลักสูตร'] = df[cfg['id_col']].astype(str).str.strip()

    # ดึงคะแนนดิบเก็บไว้ให้หน้าเว็บ (แทนที่ฝั่ง server.js)
    for p, m in zip(df['รหัสหลักสูตร'], df['คะแนนต่ำสุด']):
        if pd.notna(m) and float(m) > 0:
            min_lookup_raw[year][str(p).strip()] = float(m)

    det_col = cfg['det_col']
    df['criteria'] = df[det_col].replace('0', np.nan) if det_col in df.columns else np.nan
    fac = df[cfg['fac_col']] if cfg['fac_col'] in df.columns else np.nan
    ins = df[cfg['ins_col']] if cfg['ins_col'] in df.columns else np.nan
    app_s = pd.to_numeric(df[cfg['app_col']], errors='coerce') if cfg['app_col'] in df.columns else np.nan
    acc_s = pd.to_numeric(df[cfg['acc_col']], errors='coerce') if cfg['acc_col'] in df.columns else np.nan

    chunk = pd.DataFrame({
        'program_id':  df['รหัสหลักสูตร'],
        'faculty':     fac,
        'institution': ins,
        'min_pct':     df['min_pct'],
        'criteria':    df['criteria'],
        'applied':     app_s,
        'accepted':    acc_s,
        'year':        year,
    })
    all_data.append(chunk)

df_main = pd.concat(all_data, ignore_index=True)
df_main = df_main[(df_main['min_pct'] > 0)].dropna(subset=['min_pct', 'program_id'])

df_main['applied']   = pd.to_numeric(df_main['applied'],  errors='coerce')
df_main['accepted']  = pd.to_numeric(df_main['accepted'], errors='coerce')
df_main['comp_rate'] = df_main['applied'] / df_main['accepted'].replace(0, np.nan)

df_agg = df_main.groupby(['program_id', 'year', 'criteria'], dropna=False).agg(
    faculty     = ('faculty',    'first'),
    institution = ('institution','first'),
    min_pct     = ('min_pct',    'min'),
    comp_rate   = ('comp_rate',  'mean'),
).reset_index()

df_agg_temp = df_agg.copy()
df_agg_temp['criteria_fill'] = df_agg_temp['criteria'].fillna(NAN_PLACEHOLDER)
pivot_min = df_agg_temp.pivot_table(index=['program_id', 'criteria_fill'], columns='year', values='min_pct')

# ------------------------------------------------------------------
# 1C. โหลด CSV หลัก และสร้างฐานข้อมูลส่งหน้าเว็บแบบ Node.js
# ------------------------------------------------------------------
csv_path = find_file('tcas_round3_full_data.csv')
if csv_path is None:
    raise FileNotFoundError("❌ ไม่พบ tcas_round3_full_data.csv")

tcas_main_df = safe_read_csv(csv_path)
tcas_main_df.columns = tcas_main_df.columns.str.strip()

# แปลงทุกคอลัมน์ที่สำคัญให้เป็น String และจัดการค่าว่าง เพื่อป้องกัน Frontend พัง (.split)
tcas_main_df['program_id'] = tcas_main_df['program_id'].astype(str).str.strip()
tcas_main_df['university_name'] = tcas_main_df['university_name'].fillna("").astype(str).str.strip()
tcas_main_df['faculty_name'] = tcas_main_df['faculty_name'].fillna("").astype(str).str.strip()
tcas_main_df['program_name'] = tcas_main_df['program_name'].fillna("").astype(str).str.strip()
tcas_main_df['link'] = tcas_main_df['link'].fillna("#").astype(str).str.strip()

def parse_criteria_weights(x):
    if pd.isna(x) or not str(x).strip(): return {}
    try: return ast.literal_eval(str(x))
    except: return {}

tcas_main_df['criteria_dict'] = tcas_main_df['scores_criteria'].apply(parse_criteria_weights)
criteria_lookup = dict(zip(tcas_main_df['program_id'], tcas_main_df['criteria_dict']))

# สร้างฐานข้อมูลสำหรับหน้าเว็บ (tcasDatabase) แบบรัดกุม 100%
tcas_database_frontend = []
for _, row in tcas_main_df.iterrows():
    pid = str(row.get('program_id', '')).strip()
    if not pid or pid == 'nan' or pid == "": continue
    
    crit_str = str(row.get('scores_criteria', '{}')).replace("'", '"')
    try: criteria = json.loads(crit_str)
    except: criteria = {}
        
    try: gpax = float(row.get('min_gpax', 0))
    except: gpax = 0.0
        
    tcas_database_frontend.append({
        "id": pid,
        "uni": str(row.get('university_name', '')),
        "group": str(row.get('faculty_name', '')),
        "program": str(row.get('program_name', '')),
        "gpax_min": gpax if pd.notna(gpax) else 0.0,
        "min65": min_lookup_raw[65].get(pid, None),
        "min66": min_lookup_raw[66].get(pid, None),
        "min67": min_lookup_raw[67].get(pid, None),
        "min68": min_lookup_raw[68].get(pid, None),
        "criteria": criteria,
        "link": str(row.get('link', '#'))
    })
print(f"🟢 สร้างฐานข้อมูล Frontend เสร็จ! ทั้งหมด {len(tcas_database_frontend)} หลักสูตร")

# ------------------------------------------------------------------
# 1D. โหลดสถิติข้อสอบ A-Level + TGAT/TPAT
# ------------------------------------------------------------------
CODE_MAP = {90:'tgat', 91:'tgat1', 92:'tgat2', 93:'tgat3', 20:'tpat2', 30:'tpat3', 40:'tpat4', 50:'tpat5'}
def code_to_key(c):
    if c in CODE_MAP: return CODE_MAP[c]
    if 61 <= c <= 89: return f'a_lv_{c}'
    return None

exam_rows = []
for fname, code_col, mean_col in [('alevel.xlsx', 'รหัส', 'เฉลี่ย (Mean)'), ('Tgat-Tpat.xlsx', 'รหัสวิชา', 'เฉลี่ย (Mean)')]:
    fpath = find_file(fname)
    if not fpath: continue
    df_ex = pd.read_excel(fpath)
    for _, row in df_ex.iterrows():
        yr   = int(row['ปี']) - 2500
        code = int(row[code_col])
        key  = code_to_key(code)
        mean = pd.to_numeric(row.get(mean_col, row.get('เฉลี่ย')), errors='coerce')
        if key and pd.notna(mean):
            exam_rows.append({'year': yr, 'key': key, 'mean': float(mean)})

df_exam = pd.DataFrame(exam_rows)
exam_feature_means = df_exam.groupby('key')['mean'].mean().to_dict() if not df_exam.empty else {}
exam_pivot = df_exam.pivot_table(index='year', columns='key', values='mean') if not df_exam.empty else pd.DataFrame()


# =====================================================================
# ส่วนที่ 2: ฟังก์ชันโมเดลทำนายคะแนน (เดิม ไม่แก้ไข)
# =====================================================================
def weighted_exam_score(program_id: str, year: int) -> float:
    weights = criteria_lookup.get(program_id, {})
    if not weights: return np.nan
    year_means = exam_pivot.loc[year] if year in exam_pivot.index else pd.Series(dtype=float)
    wsum, total_w, has_any = 0.0, 0.0, False
    for subj_key, w in weights.items():
        if subj_key in year_means.index and not np.isnan(year_means[subj_key]):
            m = year_means[subj_key]
            has_any = True
        else:
            m = exam_feature_means.get(subj_key, np.nan)
            if np.isnan(m): continue
        wsum += m * w
        total_w += w
    if total_w == 0 or not has_any: return np.nan
    return wsum / total_w

def predict_min_score(program_id: str) -> Optional[float]:
    pid = str(program_id).strip()
    matching_keys = [(p, c) for (p, c) in pivot_min.index if p == pid]
    if not matching_keys: return None

    best_key = None
    for key in matching_keys:
        row = pivot_min.loc[key]
        years_available = set(col for col in REQUIRED_YEARS if col in row.index and not np.isnan(row[col]))
        if years_available == REQUIRED_YEARS:
            best_key = key
            break

    if best_key is None: return None
    min_row = pivot_min.loc[best_key]
    sorted_years = sorted(list(REQUIRED_YEARS))

    train_rows = []
    for i in range(WINDOW, len(sorted_years)):
        t, lag1, lag2 = sorted_years[i], sorted_years[i - 1], sorted_years[i - 2]
        min_l1, min_l2, tgt = min_row.get(lag1, np.nan), min_row.get(lag2, np.nan), min_row.get(t, np.nan)
        exam_sc = weighted_exam_score(pid, t)
        if any(np.isnan(v) for v in [min_l1, min_l2, tgt, exam_sc]): return None
        train_rows.append({'min_lag1': min_l1, 'min_lag2': min_l2, 'exam_score': exam_sc, 'target': tgt})

    last_yr, prev_yr, pred_yr = sorted_years[-1], sorted_years[-2], sorted_years[-1] + 1
    min_l1_pred, min_l2_pred = min_row.get(last_yr, np.nan), min_row.get(prev_yr, np.nan)
    exam_sc_pred = weighted_exam_score(pid, pred_yr)

    if any(np.isnan(v) for v in [min_l1_pred, min_l2_pred, exam_sc_pred]): return None

    X_train = np.array([[r['min_lag1'], r['min_lag2'], r['exam_score']] for r in train_rows])
    y_train = np.array([r['target'] for r in train_rows])
    x_pred = np.array([[min_l1_pred, min_l2_pred, exam_sc_pred]])

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = float(model.predict(x_pred)[0])
    return round(max(0.0, min(100.0, pred)), 2)


# =====================================================================
# ส่วนที่ 3: API Endpoints (รวมจาก Node.js มาที่นี่หมดแล้ว)
# =====================================================================

# 1. แทนที่การเรียกข้อมูลทั้งหมดให้หน้าเว็บ
@app.get("/api/database")
def get_database():
    return tcas_database_frontend

# 2. เพิ่มระบบค้นหาให้หน้าเว็บ (โคลนมาจาก server.js เป๊ะๆ)
@app.get("/api/search")
def search_database(q: str = ""):
    query = q.lower().strip()
    if len(query) < 2:
        return []
    
    results = [
        item for item in tcas_database_frontend
        if query in item['uni'].lower() or
           query in item['program'].lower() or
           query in item['group'].lower()
    ]
    return results[:50]

# 3. เพิ่มระบบดึงรายชื่อคณะ
@app.get("/api/groups")
def get_groups():
    groups = list(set(item['group'] for item in tcas_database_frontend if item['group']))
    return groups

@app.get("/api/predict/{program_id}")
def get_prediction(program_id: str):
    score = predict_min_score(program_id)
    if score is None:
        return {"program_id": program_id, "predicted_score": 0.0, "status": "no_data"}
    return {"program_id": program_id, "predicted_score": score, "status": "success"}

class PredictRequest(BaseModel):
    program_id: str

@app.post("/predict")
def post_prediction(req: PredictRequest):
    score = predict_min_score(req.program_id)
    if score is None:
        return {"program_id": req.program_id, "predicted_score": 0.0, "status": "no_data"}
    return {"program_id": req.program_id, "predicted_score": score, "status": "success"}


# =====================================================================
# ส่วนที่ 4: AI Endpoints (Claude)
# =====================================================================
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

def call_claude(system_prompt: str, user_prompt: str) -> str:
    if ai_client is None: return "⚠️ ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY — AI ยังใช้งานไม่ได้"
    try:
        msg = ai_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "\n".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    except Exception as e:
        print(f"❌ Claude error: {e}")
        return "❌ ไม่สามารถวิเคราะห์ได้ในขณะนี้"

class RankItem(BaseModel):
    rank: int
    uni: str
    program: str
    score: Any = None
    min68: Any = None
    pred: Any = None

class AIRankRequest(BaseModel):
    items: List[RankItem]
    lockFirst: bool

@app.post("/api/ai-rank")
def ai_rank(req: AIRankRequest):
    items_text = "\n".join(f"{it.rank}. {it.uni} - {it.program} | คะแนนน้อง: {it.score} | Min ปีก่อน: {it.min68 or '—'} | 🔮 ทำนาย: {it.pred or '—'}" for it in req.items)
    lock = "ล็อกอันดับ 1 ไว้" if req.lockFirst else "จัดทั้ง 10 อันดับใหม่ตามกลยุทธ์ปลอดภัย"
    advice = call_claude("คุณเป็นที่ปรึกษา TCAS ตอบภาษาไทย", f"ลิสต์คณะ:\n{items_text}\nเงื่อนไข: {lock}")
    return {"advice": advice}

class AIRecommendRequest(BaseModel):
    category: str
    source: str
    score: Optional[Dict[str, Any]] = None

@app.post("/api/ai-recommend")
def ai_recommend(req: AIRecommendRequest):
    advice = call_claude("คุณเป็นที่ปรึกษาแนะแนว TCAS", f"กลุ่มคณะ: {req.category}\nคะแนน: {req.score}")
    return {"advice": advice}

class SubjectFocusItem(BaseModel):
    key: str
    name: str
    weight: float
    score: float

class AIFocusRequest(BaseModel):
    subjects: List[SubjectFocusItem]
    dreamName: str

@app.post("/api/ai-focus")
def ai_focus(req: AIFocusRequest):
    subj_text = "\n".join(f"- {s.name}: น้ำหนัก {s.weight:.1f}% | คะแนน: {s.score}" for s in req.subjects)
    advice = call_claude("คุณเป็นที่ปรึกษาเตรียมสอบ TCAS", f"ชุดคณะ: {req.dreamName}\nวิชาสำคัญ:\n{subj_text}")
    return {"advice": advice}

# =====================================================================
# Serve หน้าเว็บ index.html (แก้ไขใหม่เพื่อความชัวร์ 100%)
# =====================================================================
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    # กำหนดตำแหน่งไฟล์ให้แม่นยำที่สุดบน Render
    path = os.path.join(current_dir, "index.html")
    if not os.path.exists(path):
        path = "index.html" # เผื่อกรณีรันจาก root directory
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"❌ เซิร์ฟเวอร์รันได้ แต่หาไฟล์ index.html ไม่เจอในระบบ: {str(e)}"