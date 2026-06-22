from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import os, ast
from sklearn.linear_model import LinearRegression

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
# 1A. โหลด Excel ปี 65-68 → สร้าง df_agg เหมือน notebook
#     key สำคัญ: (program_id, criteria) โดย criteria = คอลัมน์ "รายละเอียด"
# ------------------------------------------------------------------
all_data = []

YEAR_CONFIG = {
    65: {
        'file':    'TCAS65_maxmin.xlsx',
        'id_col':  'program_id',          # ชื่อคอลัมน์ program_id ในไฟล์ปี 65
        'min_col': None,                   # จะ handle ด้านล่าง
        'det_col': 'project_name_th',     # คอลัมน์ "รายละเอียด" ปี 65
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

    # จัดการคอลัมน์คะแนนต่ำสุดตามปี
    if year == 65:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด หลังประมวลผลรอบ 2'].fillna(df['คะแนนต่ำสุด'])
        df['รหัสหลักสูตร'] = df['program_id']
    elif year == 67:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด หลังประมวลผลรอบ 2'].fillna(df['คะแนนต่ำสุด'])
    elif year == 68:
        df['คะแนนต่ำสุด'] = df['คะแนนต่ำสุด ประมวลผลครั้งที่ 2'].fillna(df['คะแนนต่ำสุด ประมวลผลครั้งที่ 1'])

    df['min_pct'] = df['คะแนนต่ำสุด'] / MAX_NEW * 100

    # ทำความสะอาด program_id
    df['รหัสหลักสูตร'] = df[cfg['id_col']].astype(str).str.strip()

    # รายละเอียด (criteria ใน notebook) — แทน '0' ด้วย NaN เหมือน notebook
    det_col = cfg['det_col']
    if det_col in df.columns:
        df['criteria'] = df[det_col].replace('0', np.nan)
    else:
        df['criteria'] = np.nan

    # faculty / institution
    fac = df[cfg['fac_col']] if cfg['fac_col'] in df.columns else np.nan
    ins = df[cfg['ins_col']] if cfg['ins_col'] in df.columns else np.nan

    # applied / accepted
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
    print(f"✅ โหลดปี {year}: {len(chunk)} แถว")

df_main = pd.concat(all_data, ignore_index=True)
df_main = df_main[(df_main['min_pct'] > 0)].dropna(subset=['min_pct', 'program_id'])
print(f"📦 รวมทั้งหมด: {len(df_main)} แถว, {df_main['program_id'].nunique()} หลักสูตร")

# สร้าง comp_rate แล้ว aggregate ด้วย (program_id, year, criteria) เหมือน notebook
df_main['applied']   = pd.to_numeric(df_main['applied'],  errors='coerce')
df_main['accepted']  = pd.to_numeric(df_main['accepted'], errors='coerce')
df_main['comp_rate'] = df_main['applied'] / df_main['accepted'].replace(0, np.nan)

df_agg = df_main.groupby(['program_id', 'year', 'criteria'], dropna=False).agg(
    faculty     = ('faculty',    'first'),
    institution = ('institution','first'),
    min_pct     = ('min_pct',    'min'),
    comp_rate   = ('comp_rate',  'mean'),
).reset_index()

print(f"📊 df_agg: {len(df_agg)} แถว, {df_agg['program_id'].nunique()} หลักสูตร")

# ------------------------------------------------------------------
# 1B. Pivot สำหรับ lookup คะแนนและ comp_rate
#     index = (program_id, criteria_with_placeholder)
# ------------------------------------------------------------------
df_agg_temp = df_agg.copy()
df_agg_temp['criteria_fill'] = df_agg_temp['criteria'].fillna(NAN_PLACEHOLDER)

# pivot คะแนนต่ำสุดแยกตาม (program_id, criteria)
pivot_min = df_agg_temp.pivot_table(
    index=['program_id', 'criteria_fill'],
    columns='year',
    values='min_pct',
)

# pivot comp_rate
pivot_comp = df_agg_temp.pivot_table(
    index=['program_id', 'criteria_fill'],
    columns='year',
    values='comp_rate',
)

print(f"🗂️ pivot_min: {len(pivot_min)} คู่ (program_id, criteria)")

# ------------------------------------------------------------------
# 1C. โหลด CSV หลัก → criteria_lookup (program_id → score weights)
#     ใช้สำหรับคำนวณ exam_score
# ------------------------------------------------------------------
csv_path = find_file('tcas_round3_full_data.csv')
if csv_path is None:
    raise FileNotFoundError("❌ ไม่พบ tcas_round3_full_data.csv")

tcas_main_df = safe_read_csv(csv_path)
tcas_main_df['program_id'] = tcas_main_df['program_id'].astype(str).str.strip()

def parse_criteria_weights(x):
    """แปลง "{'tgat': 20, 'tpat3': 30}" → {'tgat': 20, 'tpat3': 30}"""
    if pd.isna(x):
        return {}
    try:
        return ast.literal_eval(str(x))
    except:
        return {}

tcas_main_df['criteria_dict'] = tcas_main_df['scores_criteria'].apply(parse_criteria_weights)
criteria_lookup = dict(zip(tcas_main_df['program_id'], tcas_main_df['criteria_dict']))
print(f"✅ โหลด CSV หลัก: {len(tcas_main_df)} หลักสูตร")

# ------------------------------------------------------------------
# 1D. โหลดสถิติข้อสอบ A-Level + TGAT/TPAT
#     exam_stats[year][subject_key] = mean
# ------------------------------------------------------------------
CODE_MAP = {
    90:'tgat', 91:'tgat1', 92:'tgat2', 93:'tgat3',
    20:'tpat2', 30:'tpat3', 40:'tpat4', 50:'tpat5',
}

def code_to_key(c):
    if c in CODE_MAP: return CODE_MAP[c]
    if 61 <= c <= 89: return f'a_lv_{c}'
    return None

exam_rows = []
for fname, code_col, mean_col in [
    ('alevel.xlsx',    'รหัส',     'เฉลี่ย (Mean)'),
    ('Tgat-Tpat.xlsx', 'รหัสวิชา', 'เฉลี่ย (Mean)'),
]:
    fpath = find_file(fname)
    if not fpath:
        continue
    df = pd.read_excel(fpath)
    for _, row in df.iterrows():
        yr   = int(row['ปี']) - 2500        # 2566 → 66
        code = int(row[code_col])
        key  = code_to_key(code)
        mean = pd.to_numeric(row.get(mean_col, row.get('เฉลี่ย')), errors='coerce')
        if key and pd.notna(mean):
            exam_rows.append({'year': yr, 'key': key, 'mean': float(mean)})
    print(f"✅ โหลด {fname}")

df_exam = pd.DataFrame(exam_rows)
# ค่าเฉลี่ยรวมทุกปี ใช้ fallback ถ้าปีนั้นไม่มีข้อมูล
exam_feature_means = df_exam.groupby('key')['mean'].mean().to_dict()
# pivot: year × key → mean
exam_pivot = df_exam.pivot_table(index='year', columns='key', values='mean')
print(f"📊 exam_stats ปีที่มี: {sorted(df_exam['year'].unique())}")


# =====================================================================
# ส่วนที่ 2: ฟังก์ชันโมเดลทำนายคะแนน (Strict Mode เฉพาะเจาะจงรายคณะและเกณฑ์)
# =====================================================================

def weighted_exam_score(program_id: str, year: int) -> float:
    """
    คำนวณ exam_score ถ่วงน้ำหนักเฉพาะของคณะนั้น ปีนั้น
    เหมือน weighted_exam_features() ใน notebook

    ตัวอย่าง: วิศวะจุฬา criteria = {tgat:20, tpat3:40, a_lv_61:20, a_lv_64:20}
    exam_score = (mean_tgat×20 + mean_tpat3×40 + mean_alv61×20 + mean_alv64×20) / 100
    """
    weights = criteria_lookup.get(program_id, {})
    if not weights:
        return np.nan

    # ดึงค่าเฉลี่ยข้อสอบของปีนั้น
    year_means = exam_pivot.loc[year] if year in exam_pivot.index else pd.Series(dtype=float)

    wsum, total_w, has_any = 0.0, 0.0, False
    for subj_key, w in weights.items():
        if subj_key in year_means.index and not np.isnan(year_means[subj_key]):
            m = year_means[subj_key]
            has_any = True
        else:
            # fallback: ใช้ค่าเฉลี่ยรวมทุกปี
            m = exam_feature_means.get(subj_key, np.nan)
            if np.isnan(m):
                continue
        wsum    += m * w
        total_w += w

    if total_w == 0 or not has_any:
        return np.nan
    return wsum / total_w


def predict_min_score(program_id: str) -> Optional[float]:
    """
    ทำนายคะแนนต่ำสุดปี 69 ของหลักสูตรแบบเจาะจงเฉพาะกลุ่มเกณฑ์รหัสเดียวกันเท่านั้น
    - ต้องมีข้อมูลประวัติคะแนนครบทั้ง 4 ปี (65, 66, 67, 68) เปี๊ยบ ไม่มีข้อยกเว้น
    - ใช้เฉพาะฟีเจอร์: คะแนนย้อนหลัง 1 ปี, คะแนนย้อนหลัง 2 ปี และ ความยากของข้อสอบเกณฑ์คณะนั้นๆ [min_lag1, min_lag2, exam_score]
    - ไม่ใช้อัตราการแข่งขัน (comp_rate) และไม่มีระบบสุ่มหรือ Fallback เด็ดขาด
    """
    pid = str(program_id).strip()

    # ค้นหาทุกรายละเอียดเกณฑ์ (criteria) ที่จับคู่กับ program_id นี้ใน index ของ pivot_min
    matching_keys = [
        (p, c) for (p, c) in pivot_min.index
        if p == pid
    ]

    if not matching_keys:
        print(f"❌ [Strict Mode] '{pid}' -> ไม่พบประวัติข้อมูลในฐานข้อมูล")
        return None

    # ตรวจสอบหาเกณฑ์รายละเอียดที่ข้อมูล 'ครบ 4 ปีเต็ม' เท่านั้น
    best_key = None
    for key in matching_keys:
        row = pivot_min.loc[key]
        years_available = set(col for col in REQUIRED_YEARS if col in row.index and not np.isnan(row[col]))
        if years_available == REQUIRED_YEARS:
            best_key = key
            break  # เมื่อเจอรายละเอียดที่ระบุตรงครบ 4 ปีแล้ว ให้ล็อกตัวนี้ทันที

    # หากรายละเอียดเกณฑ์นั้นข้อมูลไม่ครบ 4 ปี -> ปฏิเสธการทำนายทันที ไม่มีการเดาสุ่ม
    if best_key is None:
        print(f"❌ [Strict Mode] '{pid}' -> ข้อมูลประวัติเกณฑ์นี้มีไม่ครบ 4 ปีเต็ม จึงไม่ทำการทำนายผล")
        return None

    crit_fill = best_key[1]
    crit_label = 'NaN' if crit_fill == NAN_PLACEHOLDER else crit_fill
    print(f"🎯 [Strict Mode] กำลังเรียนรู้พฤติกรรมคะแนนของรหัส '{pid}' เกณฑ์: '{str(crit_label)[:50]}'")

    min_row = pivot_min.loc[best_key]
    sorted_years = sorted(list(REQUIRED_YEARS))  # จะได้ลำดับเวลา [65, 66, 67, 68]

    # ------------------------------------------------------------------
    # ขั้นตอนการทำตารางฝึกสอน (Training Dataset) ด้วย Sliding Window
    # ------------------------------------------------------------------
    train_rows = []
    for i in range(WINDOW, len(sorted_years)):
        t     = sorted_years[i]          # ปีผลลัพธ์ (เช่น 67, 68)
        lag1  = sorted_years[i - 1]      # ย้อนหลัง 1 ปี
        lag2  = sorted_years[i - 2]      # ย้อนหลัง 2 ปี

        min_lag1 = min_row.get(lag1, np.nan)
        min_lag2 = min_row.get(lag2, np.nan)
        target   = min_row.get(t, np.nan)
        
        # ค้นหาค่าน้ำหนักความยากง่ายข้อสอบของเกณฑ์วิชาคณะนี้ ณ ปีเป้าหมายนั้นๆ
        exam_sc  = weighted_exam_score(pid, t)

        # หากมีค่าสูญหายหรือ NaN แม้แต่จุดเดียวในชุดหน้าต่างเวลานี้ จะสั่งปิดการทำนายทันที
        if any(np.isnan(v) for v in [min_lag1, min_lag2, target, exam_sc]):
            print(f"❌ [Strict Mode] สถิติประวัติช่วงปี {t} หรือข้อมูลคะสอบเฉลี่ยของเกณฑ์ไม่สมบูรณ์ -> ปฏิเสธการทำงาน")
            return None

        train_rows.append({
            'min_lag1':   min_lag1,
            'min_lag2':   min_lag2,
            'exam_score': exam_sc,
            'target':     target,
        })

    # ------------------------------------------------------------------
    # เตรียมชุดตัวแปรอินพุตสำหรับพยากรณ์คะแนนของ "ปี 69"
    # ------------------------------------------------------------------
    last_year = sorted_years[-1]   # 68
    prev_year = sorted_years[-2]   # 67
    pred_year = last_year + 1      # 69

    min_lag1_pred = min_row.get(last_year, np.nan)
    min_lag2_pred = min_row.get(prev_year, np.nan)
    exam_sc_pred  = weighted_exam_score(pid, pred_year)  # ดึงค่าเฉลี่ยข้อสอบที่เพิ่งสอบไปสำหรับปี 69

    # ตรวจเช็กความสมบูรณ์ของตัวแปรอินพุตก่อนส่งให้สมการทำนายปี 69
    if any(np.isnan(v) for v in [min_lag1_pred, min_lag2_pred, exam_sc_pred]):
        print(f"❌ [Strict Mode] อินพุตใช้พยากรณ์ปี 69 ข้อมูลไม่ครบถ้วน -> ปฏิเสธการทำงาน")
        return None

    # ------------------------------------------------------------------
    # ประมวลผลโมเดล Linear Regression คณะใครคณะมัน 100%
    # ------------------------------------------------------------------
    features = ['min_lag1', 'min_lag2', 'exam_score']
    X_train = np.array([[r[f] for f in features] for r in train_rows])
    y_train = np.array([r['target'] for r in train_rows])
    x_pred  = np.array([[min_lag1_pred, min_lag2_pred, exam_sc_pred]])

    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # คำนวณผลและตีกรอบเปอร์เซ็นต์ให้อยู่ในช่วงคะแนน 0 - 100 เสมอ
    pred = float(model.predict(x_pred)[0])
    pred = round(max(0.0, min(100.0, pred)), 2)
    
    print(f"🔮 [Prediction Success] คะแนนต่ำสุดทำนายปี 69 ของเกณฑ์นี้คือ: {pred}")
    return pred

# =====================================================================
# ส่วนที่ 2.5: ประเมินประสิทธิภาพโมเดลด้วย Leave-One-Out (ทดสอบทำนายปี 68)
#   - เปรียบเทียบ RMSE และ MAE กับ 2 baseline:
#       baseline1 = ใช้คะแนนปีก่อน (lag1)
#       baseline2 = ค่าเฉลี่ย 3 ปีย้อนหลัง
# =====================================================================
def evaluate_model():
    errors_model     = []   # สำหรับ RMSE (ยกกำลังสอง)
    abs_errors_model = []   # สำหรับ MAE (ค่าสัมบูรณ์)
    errors_baseline1 = []
    abs_errors_b1    = []
    errors_baseline2 = []
    abs_errors_b2    = []
    case_details     = []

    for (pid, crit), row in pivot_min.iterrows():
        years_available = [y for y in ALL_YEARS if y in row.index and not np.isnan(row[y])]
        if set(years_available) != REQUIRED_YEARS:
            continue

        test_year   = 68
        train_years = [65, 66, 67]

        train_rows = []
        for i in range(WINDOW, len(train_years)):
            lag1    = row[train_years[i - 1]]
            lag2    = row[train_years[i - 2]]
            target  = row[train_years[i]]
            exam_sc = weighted_exam_score(pid, train_years[i])
            if any(np.isnan(v) for v in [lag1, lag2, target, exam_sc]):
                break
            train_rows.append([lag1, lag2, exam_sc, target])

        if len(train_rows) < 1:
            continue

        try:
            X = np.array([r[:3] for r in train_rows])
            y = np.array([r[3]  for r in train_rows])
            model = LinearRegression().fit(X, y)

            lag1_pred = float(row[67])
            lag2_pred = float(row[66])
            exam_pred = weighted_exam_score(pid, test_year)
            if any(np.isnan(v) for v in [lag1_pred, lag2_pred, exam_pred]):
                continue

            pred   = float(model.predict([[lag1_pred, lag2_pred, exam_pred]])[0])
            actual = float(row[test_year])
            error  = pred - actual

            errors_model.append(error ** 2)
            abs_errors_model.append(abs(error))           # เพิ่ม
            errors_baseline1.append((float(row[67]) - actual) ** 2)
            abs_errors_b1.append(abs(float(row[67]) - actual))    # เพิ่ม
            errors_baseline2.append((float(np.mean([row[65], row[66], row[67]])) - actual) ** 2)
            abs_errors_b2.append(abs(float(np.mean([row[65], row[66], row[67]])) - actual))  # เพิ่ม)

            # ดึงข้อมูลเพิ่มเติมจาก tcas_main_df
            prog_row = tcas_main_df[tcas_main_df['program_id'] == pid]
            uni_name  = str(prog_row['university_name'].values[0]) if len(prog_row) else pid
            fac_name  = str(prog_row['faculty_name'].values[0])    if len(prog_row) else ''
            prog_name = str(prog_row['program_name'].values[0])    if len(prog_row) else ''
            try:
                crit_w = ast.literal_eval(str(prog_row['scores_criteria'].values[0])) if len(prog_row) else {}
            except:
                crit_w = {}

            case_details.append({
                "program_id":  pid,
                "uni":         uni_name,
                "faculty":     fac_name,
                "program":     prog_name,
                "criteria":    crit_w,
                "min65":       round(float(row[65]), 2) if not np.isnan(row[65]) else None,
                "min66":       round(float(row[66]), 2) if not np.isnan(row[66]) else None,
                "min67":       round(float(row[67]), 2) if not np.isnan(row[67]) else None,
                "min68":       round(float(row[68]), 2) if not np.isnan(row[68]) else None,
                "predicted":   round(pred, 2),
                "actual":      round(actual, 2),
                "error":       round(error, 2),
                "abs_error":   round(abs(error), 2),
            })
        except:
            continue

    result = {}
    if errors_model:
        rmse_model     = float(np.sqrt(np.mean(errors_model)))
        rmse_baseline1 = float(np.sqrt(np.mean(errors_baseline1)))
        rmse_baseline2 = float(np.sqrt(np.mean(errors_baseline2)))
        mae_model      = float(np.mean(abs_errors_model))      # เพิ่ม
        mae_baseline1  = float(np.mean(abs_errors_b1))         # เพิ่ม
        mae_baseline2  = float(np.mean(abs_errors_b2))         # เพิ่ม

        print(f"RMSE โมเดล:            {rmse_model:.2f}%")
        print(f"MAE  โมเดล:            {mae_model:.2f}%")
        print(f"RMSE baseline ปีก่อน:  {rmse_baseline1:.2f}%")
        print(f"MAE  baseline ปีก่อน:  {mae_baseline1:.2f}%")

        case_details.sort(key=lambda x: x["abs_error"], reverse=True)
        over  = [x for x in case_details if x["error"] > 0]
        under = [x for x in case_details if x["error"] < 0]

        result = {
            "rmse_model":         round(rmse_model, 2),
            "mae_model":          round(mae_model, 2),  
            "rmse_baseline_lag1": round(rmse_baseline1, 2),
            "mae_baseline_lag1":  round(mae_baseline1, 2),  
            "rmse_baseline_avg3": round(rmse_baseline2, 2),
            "mae_baseline_avg3":  round(mae_baseline2, 2),  
            "better_than_lag1_rmse": round(rmse_baseline1 - rmse_model, 2),
            "better_than_lag1_mae":  round(mae_baseline1 - mae_model, 2),  
            "better_than_avg3_rmse": round(rmse_baseline2 - rmse_model, 2),
            "better_than_avg3_mae":  round(mae_baseline2 - mae_model, 2), 
            "n_programs":          len(errors_model),
            "over_predict_count":  len(over),
            "under_predict_count": len(under),
            "worst_10": case_details[:10],
            "best_10":  case_details[-10:][::-1],
        }

    return result

eval_result = evaluate_model()

# คำนวณ risk threshold รายหลักสูตร จาก SD ของ Min ข้ามปี
def compute_risk_thresholds():
    result = {}
    for (pid, crit), row in pivot_min.iterrows():
        years_available = [y for y in ALL_YEARS if y in row.index and not np.isnan(row[y])]
        if len(years_available) < 2:
            continue
        scores = [float(row[y]) for y in years_available]
        diffs  = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        mean_d = float(np.mean(diffs))
        sd_d   = float(np.std(diffs)) if len(diffs) > 1 else 3.0
        if sd_d < 1.0:
            sd_d = 1.0  # กันไม่ให้ SD เล็กเกินไป
        if pid not in result:
            result[pid] = {"mean_diff": round(mean_d, 2), "sd_diff": round(sd_d, 2)}
    return result

risk_thresholds = compute_risk_thresholds()
print(f"📊 คำนวณ risk threshold สำเร็จ: {len(risk_thresholds)} หลักสูตร")

@app.get("/api/risk-thresholds")
def get_risk_thresholds():
    return risk_thresholds

@app.get("/api/model-eval")
def get_model_eval():
    return eval_result

@app.get("/api/error-analysis")
def get_error_analysis():
    return {
        "worst_predictions": eval_result.get("worst_10", []),
        "best_predictions":  eval_result.get("best_10", []),
        "over_predict_count":  eval_result.get("over_predict_count", 0),
        "under_predict_count": eval_result.get("under_predict_count", 0),
        "summary": (
            "โมเดล over-predict มากกว่า" if eval_result.get("over_predict_count", 0) > eval_result.get("under_predict_count", 0)
            else "โมเดล under-predict มากกว่า"
        )
    }

# =====================================================================
# ส่วนที่ 3: API Endpoints
# =====================================================================

@app.get("/api/database")
def get_database():
    # สร้าง lookup คะแนนต่ำสุดแต่ละปี จาก df_agg
    min_by_year = {}
    for year in [65, 66, 67, 68]:
        year_df = df_agg[df_agg['year'] == year]
        min_by_year[year] = dict(zip(
            year_df['program_id'],
            year_df['min_pct']
        ))

    result = []
    for _, row in tcas_main_df.iterrows():
        try:
            criteria = ast.literal_eval(str(row.get("scores_criteria", "{}")))
        except:
            criteria = {}
        pid = str(row.get("program_id", ""))
        result.append({
            "id": pid,
            "uni": str(row.get("university_name", "")),
            "group": str(row.get("faculty_name", "")),
            "program": str(row.get("program_name", "")),
            "gpax_min": float(row.get("min_gpax") or 0),
            "min65": min_by_year[65].get(pid),
            "min66": min_by_year[66].get(pid),
            "min67": min_by_year[67].get(pid),
            "min68": min_by_year[68].get(pid),
            "criteria": criteria,
            "link": str(row.get("link", "#")),
        })
    return result

@app.get("/api/search")
def search(q: str = ""):
    if len(q) < 2:
        return []
    q_lower = q.lower()
    results = []
    for _, row in tcas_main_df.iterrows():
        if (q_lower in str(row.get("university_name","")).lower() or
            q_lower in str(row.get("program_name","")).lower() or
            q_lower in str(row.get("faculty_name","")).lower()):
            results.append({
                "id": str(row.get("program_id","")),
                "uni": str(row.get("university_name","")),
                "group": str(row.get("faculty_name","")),
                "program": str(row.get("program_name","")),
            })
        if len(results) >= 50:
            break
    return results

@app.get("/api/groups")
def get_groups():
    return tcas_main_df["faculty_name"].dropna().unique().tolist()

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
# ส่วนที่ 4: AI Endpoints (Gemini)
# =====================================================================
import urllib.request, json as _json

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def call_claude(system_prompt: str, user_prompt: str) -> str:
    """เรียก Gemini API (ใช้ชื่อ call_claude เพื่อไม่ต้องเปลี่ยน code ที่เรียกใช้)"""
    if not GEMINI_API_KEY:
        return "⚠️ ยังไม่ได้ตั้งค่า GEMINI_API_KEY — AI ยังใช้งานไม่ได้"
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={GEMINI_API_KEY}"
        body = _json.dumps({
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}]
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as res:
            data = _json.loads(res.read())
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"❌ Gemini error: {e}")
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
    items_text = "\n".join(
        f"{it.rank}. {it.uni} - {it.program} | คะแนนน้อง: {it.score} | "
        f"Min ปีก่อน: {it.min68 or '—'} | 🔮 ทำนาย: {it.pred or '—'}"
        for it in req.items
    )
    lock = (
        "ล็อกอันดับ 1 ไว้ จัดอันดับ 2–10 ใหม่ตามกลยุทธ์ปลอดภัย"
        if req.lockFirst else
        "จัดทั้ง 10 อันดับใหม่ตามกลยุทธ์ปลอดภัย (1–3 เสี่ยง, กลาง ๆ ตามตัว, ท้าย ๆ ชัวร์)"
    )
    advice = call_claude(
        "คุณเป็นที่ปรึกษา TCAS ตอบภาษาไทย กระชับ แบ่ง 2 ส่วน: "
        "1) สรุปความเสี่ยงอันดับเดิม 2) อันดับแนะนำใหม่พร้อมเหตุผล",
        f"ลิสต์คณะ 10 อันดับ:\n{items_text}\n\nเงื่อนไข: {lock}"
    )
    return {"advice": advice}


class AIRecommendRequest(BaseModel):
    category: str
    source: str
    score: Optional[Dict[str, Any]] = None

@app.post("/api/ai-recommend")
def ai_recommend(req: AIRecommendRequest):
    CAT_TH = {
        "sci": "วิทยาศาสตร์/วิศวกรรม/เทคโนโลยี",
        "arts": "มนุษยศาสตร์/สังคมศาสตร์/ครุศาสตร์",
        "biz": "บริหาร/บัญชี/เศรษฐศาสตร์/นิติ",
        "creative": "ศิลปกรรม/นิเทศ/สถาปัตย์",
    }
    score_text = "ไม่ได้ระบุ" if not req.score else ", ".join(f"{k}:{v}" for k,v in req.score.items())
    advice = call_claude(
        "คุณเป็นที่ปรึกษาแนะแนว TCAS ตอบภาษาไทย 3–5 บรรทัด",
        f"แหล่งที่มา: {req.source}\nกลุ่มคณะแนะนำ: {CAT_TH.get(req.category, req.category)}\n"
        f"คะแนน: {score_text}\nอธิบายว่าทำไมเหมาะกับน้อง"
    )
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
    subj_text = "\n".join(
        f"- {s.name}: น้ำหนัก {s.weight:.1f}% | คะแนนน้อง: {s.score}"
        for s in req.subjects
    )
    advice = call_claude(
        "คุณเป็นที่ปรึกษาเตรียมสอบ TCAS ตอบภาษาไทย แบ่งเป็นข้อ ๆ "
        "สำหรับแต่ละวิชาที่แนะนำให้โฟกัส ต้องอธิบายด้วยว่า "
        "'ความคุ้มค่าในการโฟกัส' คิดจากอะไร เช่น น้ำหนักสูงแต่คะแนนยังต่ำ หรือ คะแนนใกล้เป้าแต่น้ำหนักมาก "
        "โดยคำนวณจาก (น้ำหนัก × ช่องว่างจากเป้า) อธิบายให้น้องเข้าใจได้ง่าย",
        f"ชุดคณะ: {req.dreamName}\nวิชาและคะแนน:\n{subj_text}\n"
        f"เป้าคะแนนแต่ละวิชาคือ 80 คะแนน\n"
        f"แนะนำวิชาที่ควรโฟกัส พร้อมอธิบายเหตุผลว่าทำไมวิชานั้นถึงคุ้มค่าที่สุด"
    )
    return {"advice": advice}

@app.get("/api/eda-summary")
def eda_summary():
    # กระจายคะแนนแต่ละปี
    score_dist = {}
    for year in ALL_YEARS:
        yr_data = df_agg[df_agg['year'] == year]['min_pct'].dropna()
        score_dist[str(year)] = {
            "mean":   round(float(yr_data.mean()), 2),
            "median": round(float(yr_data.median()), 2),
            "std":    round(float(yr_data.std()), 2),
            "min":    round(float(yr_data.min()), 2),
            "max":    round(float(yr_data.max()), 2),
        }

    # correlation ระหว่าง comp_rate กับ min_pct
    merged = df_agg[['min_pct','comp_rate']].dropna()
    corr = float(merged['min_pct'].corr(merged['comp_rate']))

    return {
        "score_distribution_by_year": score_dist,
        "correlation_comprate_minscore": round(corr, 3),
        "total_programs": int(df_agg['program_id'].nunique()),
        "programs_with_all_4_years": int(
            df_agg.groupby('program_id')['year']
            .nunique().eq(4).sum()
        )
    }


# =====================================================================
# Serve static files (index.html) จาก public/
# =====================================================================
public_dir = os.path.join(current_dir, "public")
if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")