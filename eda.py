import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import ast, os, warnings
warnings.filterwarnings('ignore')

# ── ฟอนต์ภาษาไทย (รองรับ Windows และลดการแจ้งเตือน) ─────────────────
for path in [
    'C:/Windows/Fonts/tahoma.ttf',                      # Windows
    '/usr/share/fonts/truetype/thai-tlwg/TlwgMono.ttf', # Linux
    '/System/Library/Fonts/Supplemental/Tahoma.ttf',    # Mac
]:
    if os.path.exists(path):
        fm.fontManager.addfont(path)

plt.rcParams['font.family'] = ['Tahoma', 'TlwgMono', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = 'data'
MAX_NEW  = 100

# ── โหลดข้อมูล ─────────────────────────────────────────────────────
configs = {
    65: ('TCAS65_maxmin.xlsx', 'program_id',    'project_name_th',
         'คะแนนต่ำสุด หลังประมวลผลรอบ 2', 'สมัคร','รับ',
         'program_lookup_programs.faculty_name_th','university_name'),
    66: ('TCAS66_maxmin.xlsx', 'รหัสหลักสูตร', 'รายละเอียด',
         'คะแนนต่ำสุด',                         'สมัคร','รับ',
         'คณะ/สำนักวิชา','สถาบัน'),
    67: ('TCAS67_maxmin.xlsx', 'รหัสหลักสูตร', 'รายละเอียด',
         'คะแนนต่ำสุด หลังประมวลผลรอบ 2',      'สมัคร','รับ',
         'คณะ','สถาบัน'),
    68: ('TCAS68_maxmin.xlsx', 'รหัสหลักสูตร', 'รายละเอียด',
         'คะแนนต่ำสุด ประมวลผลครั้งที่ 2',      'สมัคร','รับ',
         'คณะ','สถาบัน'),
}

all_data = []
for year,(f,id_c,det_c,sc_c,app_c,acc_c,fac_c,ins_c) in configs.items():
    fp = os.path.join(DATA_DIR, f)
    if not os.path.exists(fp): continue
    df = pd.read_excel(fp)
    df['program_id'] = df[id_c].astype(str).str.strip()
    
    # จัดการรายละเอียดเกณฑ์ให้เป็นข้อความที่สะอาด เพื่อให้จับคู่ "ของใครของมัน" ได้แม่นยำ
    if det_c in df.columns:
        df['criteria'] = df[det_c].fillna('ทั่วไป').astype(str).str.strip()
        df['criteria'] = df['criteria'].replace(['0', '0.0', 'nan', 'NaN'], 'ทั่วไป')
    else:
        df['criteria'] = 'ทั่วไป'
        
    df['min_pct']    = pd.to_numeric(df[sc_c], errors='coerce') / MAX_NEW * 100
    df['applied']    = pd.to_numeric(df[app_c], errors='coerce') if app_c in df.columns else np.nan
    df['accepted']   = pd.to_numeric(df[acc_c], errors='coerce') if acc_c in df.columns else np.nan
    df['faculty']    = df[fac_c] if fac_c in df.columns else np.nan
    df['institution']= df[ins_c] if ins_c in df.columns else np.nan
    df['year']       = year
    df = df[df['min_pct'] > 0].dropna(subset=['min_pct','program_id'])
    all_data.append(df[['program_id','criteria','min_pct','applied','accepted','faculty','institution','year']])

df_all = pd.concat(all_data, ignore_index=True)
df_all['comp_rate'] = df_all['applied'] / df_all['accepted'].replace(0, np.nan)

df_agg = df_all.groupby(['program_id','year','criteria'], dropna=False).agg(
    faculty=('faculty','first'), institution=('institution','first'),
    min_pct=('min_pct','min'), comp_rate=('comp_rate','mean')
).reset_index()

print(f"โหลดเสร็จ: {len(df_agg)} แถว, {df_agg['program_id'].nunique()} หลักสูตร")

# ── FACULTY GROUP ────────────────────────────────────────────────────
FAC_MAP = {
    'วิศวกรรม': 'วิศวกรรม',
    'วิทยาศาสตร์': 'วิทยาศาสตร์',
    'แพทย์': 'แพทย์/สาธารณสุข',
    'สาธารณสุข': 'แพทย์/สาธารณสุข',
    'บริหาร': 'บริหาร/เศรษฐศาสตร์',
    'เศรษฐศาสตร์': 'บริหาร/เศรษฐศาสตร์',
    'ครุศาสตร์': 'ครุ/ศึกษาศาสตร์',
    'ศึกษาศาสตร์': 'ครุ/ศึกษาศาสตร์',
    'มนุษยศาสตร์': 'มนุษย์/ศิลปศาสตร์',
    'ศิลปศาสตร์': 'มนุษย์/ศิลปศาสตร์',
    'นิติ': 'นิติ/รัฐศาสตร์',
    'รัฐศาสตร์': 'นิติ/รัฐศาสตร์',
}
def map_fac(f):
    if pd.isna(f): return 'อื่นๆ'
    for k,v in FAC_MAP.items():
        if k in str(f): return v
    return 'อื่นๆ'

df_agg['fac_group'] = df_agg['faculty'].apply(map_fac)

COLORS = ['#C9956C','#8B6914','#D4A853','#6B8E6E','#7B9EC9','#C47B7B','#9B8BB4','#5C8A8A']
YEARS  = [65,66,67,68]

# ════════════════════════════════════════════════════════════════════
# กราฟที่ 1: แนวโน้มคะแนนเฉลี่ยรายปี รวม + แยกกลุ่มคณะ
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EDA: แนวโน้มคะแนนต่ำสุด TCAS รอบ 3 ปี 65-68', fontsize=15, fontweight='bold', y=1.01)

# ซ้าย: ภาพรวม
ax = axes[0]
trend = df_agg.groupby('year')['min_pct'].agg(['mean','median','std']).reindex(YEARS)
ax.fill_between(YEARS,
    trend['mean'] - trend['std'],
    trend['mean'] + trend['std'],
    alpha=0.15, color=COLORS[0], label='±1 SD')
ax.plot(YEARS, trend['mean'],   'o-', color=COLORS[0], lw=2.5, ms=8, label='ค่าเฉลี่ย')
ax.plot(YEARS, trend['median'], 's--', color=COLORS[1], lw=2, ms=7,  label='มัธยฐาน')
for yr, m in zip(YEARS, trend['mean']):
    ax.annotate(f'{m:.1f}', (yr, m), textcoords='offset points', xytext=(0,10), ha='center', fontsize=11)
ax.set_title('ภาพรวมทุกหลักสูตร', fontsize=13)
ax.set_xlabel('ปี TCAS'); ax.set_ylabel('คะแนนต่ำสุด (%)')
ax.set_xticks(YEARS); ax.legend(); ax.grid(alpha=0.3)

# ขวา: แยกกลุ่มคณะ
ax2 = axes[1]
fac_trend = df_agg.groupby(['year','fac_group'])['min_pct'].mean().unstack()
top_facs  = fac_trend.mean().nlargest(6).index
for i, fac in enumerate(top_facs):
    if fac in fac_trend.columns:
        ax2.plot(YEARS, fac_trend[fac].reindex(YEARS), 'o-',
                 color=COLORS[i], lw=2, ms=7, label=fac)
ax2.set_title('แยกตามกลุ่มคณะ (Top 6)', fontsize=13)
ax2.set_xlabel('ปี TCAS'); ax2.set_ylabel('คะแนนต่ำสุดเฉลี่ย (%)')
ax2.set_xticks(YEARS); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('eda_01_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก eda_01_trend.png")

# ════════════════════════════════════════════════════════════════════
# กราฟที่ 2: การกระจาย (Distribution) และความผันผวน SD
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EDA: การกระจายและความผันผวนคะแนน', fontsize=15, fontweight='bold')

# ซ้าย: Box plot แต่ละปี
ax = axes[0]
data_by_year = [df_agg[df_agg['year']==y]['min_pct'].dropna().values for y in YEARS]
bp = ax.boxplot(data_by_year, tick_labels=[f'ปี {y}' for y in YEARS],
                patch_artist=True, medianprops=dict(color='white', lw=2))
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
stats = df_agg.groupby('year')['min_pct'].agg(['mean','std','median']).reindex(YEARS)
for i, (yr, row) in enumerate(stats.iterrows()):
    ax.text(i+1, row['mean']+2, f"μ={row['mean']:.1f}\nσ={row['std']:.1f}",
            ha='center', fontsize=9, color='#333')
ax.set_title('Box Plot คะแนนต่ำสุดรายปี', fontsize=13)
ax.set_ylabel('คะแนนต่ำสุด (%)'); ax.grid(axis='y', alpha=0.3)

# ขวา: Histogram SD ของแต่ละหลักสูตรแบบแยกเกณฑ์ของใครของมัน
ax2 = axes[1]
pivot = df_agg.pivot_table(index=['program_id', 'criteria'], columns='year', values='min_pct')
pivot = pivot.dropna()
sd_per = pivot.std(axis=1)
ax2.hist(sd_per, bins=40, color=COLORS[0], alpha=0.8, edgecolor='white')
ax2.axvline(sd_per.mean(),   color=COLORS[1], lw=2.5, linestyle='--', label=f'ค่าเฉลี่ย SD = {sd_per.mean():.2f}')
ax2.axvline(sd_per.median(), color=COLORS[2], lw=2,   linestyle=':',  label=f'มัธยฐาน SD = {sd_per.median():.2f}')
ax2.set_title('Distribution ของ SD รายหลักสูตร (แยกตามเกณฑ์)', fontsize=13)
ax2.set_xlabel('SD คะแนนต่ำสุด (%)'); ax2.set_ylabel('จำนวนหลักสูตรย่อย')
ax2.legend(); ax2.grid(alpha=0.3)
print(f"SD รายหลักสูตรย่อย: mean={sd_per.mean():.2f}, median={sd_per.median():.2f}, max={sd_per.max():.2f}")

plt.tight_layout()
plt.savefig('eda_02_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก eda_02_distribution.png")

# ════════════════════════════════════════════════════════════════════
# กราฟที่ 3: Correlation ข้ามมหาลัย (คณะเดียวกัน)
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EDA: Correlation ข้ามมหาวิทยาลัย vs ข้ามคณะภายในมหาวิทยาลัย', fontsize=14, fontweight='bold')

# ซ้าย: คณะเดียวกัน ต่างมหาลัย — จับคู่แบบ "รหัส + เกณฑ์รายละเอียด" ตรงกันเป๊ะๆ อันไหนไม่มีก็ตัดทิ้ง
ax = axes[0]
for i, fac in enumerate(list(top_facs)[:4]):
    sub = df_agg[df_agg['fac_group']==fac]
    p67 = sub[sub['year']==67].set_index(['program_id', 'criteria'])['min_pct']
    p68 = sub[sub['year']==68].set_index(['program_id', 'criteria'])['min_pct']
    merged = pd.concat([p67, p68], axis=1, keys=['67','68']).dropna() # dropna จะตัดอันที่จับคู่กันไม่ได้ทิ้งทันที
    if len(merged) > 5:
        ax.scatter(merged['67'], merged['68'], alpha=0.5, s=20, color=COLORS[i], label=fac)
        corr = merged['67'].corr(merged['68'])
        print(f"  {fac}: corr(ปี67,ปี68) = {corr:.3f}  (n={len(merged)})")

lims = [0,100]
ax.plot(lims, lims, 'k--', alpha=0.3, lw=1, label='y=x')
ax.set_xlim(0,100); ax.set_ylim(0,100)
ax.set_xlabel('คะแนนต่ำสุด ปี 67 (%)'); ax.set_ylabel('คะแนนต่ำสุด ปี 68 (%)')
ax.set_title('Scatter ปี67 vs ปี68 แยกกลุ่มคณะ (จับคู่ตามรหัส+เกณฑ์)', fontsize=12)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ขวา: ภายในมหาลัยเดียวกัน — จับคู่แบบ "รหัส + เกณฑ์รายละเอียด" ตรงกันเป๊ะๆ
ax2 = axes[1]
top_unis = df_agg.groupby('institution')['program_id'].nunique().nlargest(5).index.tolist()
corr_data = []
for uni in top_unis:
    sub = df_agg[(df_agg['institution']==uni) & (df_agg['year'].isin([67,68]))]
    sub67 = sub[sub['year']==67].set_index(['program_id', 'criteria'])['min_pct']
    sub68 = sub[sub['year']==68].set_index(['program_id', 'criteria'])['min_pct']
    merged = pd.concat([sub67, sub68], axis=1, keys=['67','68']).dropna() # ตัดเกณฑ์ที่ปีใดปีหนึ่งไม่มีทิ้ง
    corr = merged['67'].corr(merged['68']) if len(merged)>3 else np.nan
    corr_data.append({'มหาวิทยาลัย': str(uni)[:12], 'corr': round(corr,3) if not np.isnan(corr) else None, 'n': len(merged)})

df_corr = pd.DataFrame(corr_data).dropna()
bars = ax2.barh(df_corr['มหาวิทยาลัย'], df_corr['corr'],
                color=[COLORS[i] for i in range(len(df_corr))], alpha=0.85)
for bar, val, n in zip(bars, df_corr['corr'], df_corr['n']):
    ax2.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
             f'{val:.3f} (n={n})', va='center', fontsize=10)
ax2.set_xlim(0, 1.15)
ax2.set_title('Correlation ปี67→68 ภายในมหาวิทยาลัย\n(สูง = คณะ/เกณฑ์ต่างๆ ขึ้นลงพร้อมกัน)', fontsize=11)
ax2.set_xlabel('Pearson Correlation'); ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_03_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก eda_03_correlation.png")

# ════════════════════════════════════════════════════════════════════
# กราฟที่ 4: YoY Change และ Outlier
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('EDA: การเปลี่ยนแปลงคะแนน YoY และ Outlier', fontsize=14, fontweight='bold')

# ซ้าย: Bar chart YoY change แต่ละกลุ่มคณะ
ax = axes[0]
yoy = {}
for fac in list(top_facs)[:6]:
    sub = df_agg[df_agg['fac_group']==fac]
    means = sub.groupby('year')['min_pct'].mean().reindex(YEARS)
    yoy[fac] = {f'{YEARS[i]}-{YEARS[i+1]}': means[YEARS[i+1]] - means[YEARS[i]] for i in range(len(YEARS)-1)}
df_yoy = pd.DataFrame(yoy).T
x = np.arange(len(df_yoy.columns))
w = 0.12
for i, (fac, row) in enumerate(df_yoy.iterrows()):
    ax.bar(x + i*w, row.values, w, label=fac, color=COLORS[i], alpha=0.85)
ax.axhline(0, color='black', lw=1)
ax.set_xticks(x + w*2.5)
ax.set_xticklabels(df_yoy.columns)
ax.set_title('การเปลี่ยนแปลงคะแนนเฉลี่ย YoY แต่ละกลุ่มคณะ', fontsize=12)
ax.set_ylabel('Δ คะแนน (%)'); ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# ขวา: หลักสูตรย่อยแยกตามเกณฑ์ที่คะแนนผันผวนมากที่สุด top 10
ax2 = axes[1]
pivot4 = df_agg.pivot_table(index=['program_id', 'criteria'], columns='year', values='min_pct').dropna()
sd4 = pivot4.std(axis=1).nlargest(10)
labels = []
for pid, crit in sd4.index:
    row = df_agg[(df_agg['program_id']==pid) & (df_agg['criteria']==crit)].iloc[0]
    uni  = str(row['institution'])[:10] if pd.notna(row['institution']) else pid
    fac  = str(row['faculty'])[:8] if pd.notna(row['faculty']) else ''
    crit_short = str(crit)[:8]
    labels.append(f"{uni}\n{fac}({crit_short})")
ax2.barh(range(len(sd4)), sd4.values[::-1], color=COLORS[0], alpha=0.85)
ax2.set_yticks(range(len(sd4)))
ax2.set_yticklabels(labels[::-1], fontsize=8)
ax2.set_title('Top 10 เกณฑ์หลักสูตรที่คะแนนผันผวนมากที่สุด', fontsize=12)
ax2.set_xlabel('SD คะแนนต่ำสุด (%)'); ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_04_yoy_outlier.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ บันทึก eda_04_yoy_outlier.png")
print("\n✅ EDA เสร็จครบ ได้ 4 รูป: eda_01_trend.png, eda_02_distribution.png, eda_03_correlation.png, eda_04_yoy_outlier.png")

import matplotlib.pyplot as plt

# 1. ระบุชื่อคณะที่อยากเปรียบเทียบ (เช่น วิศวกรรมศาสตร์)
target_faculty = "วิศวกรรมศาสตร์"
df_target = df_agg[df_agg['faculty'].astype(str).str.contains(target_faculty, na=False)]

# 2. เลือกมหาวิทยาลัยที่อยากเปรียบเทียบ 3 แห่ง
target_unis = ["จุฬาลงกรณ์มหาวิทยาลัย", "มหาวิทยาลัยเกษตรศาสตร์", "มหาวิทยาลัยเชียงใหม่"]

plt.figure(figsize=(10, 5))
for uni in target_unis:
    # กรองเอาเฉพาะมหาวิทยาลัยนั้นๆ
    df_uni = df_target[df_target['institution'].astype(str).str.contains(uni, na=False)]
    # หาค่าเฉลี่ยคะแนนต่ำสุดแยกตามปี
    trend = df_uni.groupby('year')['min_pct'].mean()
    if not trend.empty:
        plt.plot(trend.index, trend.values, marker='o', linewidth=2, label=uni)

plt.title(f'ความสัมพันธ์ของแนวโน้มคะแนนเฉลี่ยคณะ{target_faculty} ข้ามมหาวิทยาลัย', fontsize=14)
plt.xlabel('ปี TCAS (65-68)', fontsize=12)
plt.ylabel('คะแนนเฉลี่ยต่ำสุด (%)', fontsize=12)
plt.xticks([65, 66, 67, 68])
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# เซฟรูปเพื่อนำไปแปะ Medium
plt.savefig('eda_correlation_faculty.png', dpi=300)
plt.show()
# =================================================================
# ส่วนที่เพิ่มใหม่: ประเมินโมเดล (Evaluation) และ Error Analysis
# =================================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("\n" + "="*50)
print(" 📊 ผลการทดลองเปรียบเทียบโมเดล (Model Evaluation)")
print("="*50)

# ดึงข้อมูลคณะที่มีข้อมูลครบทั้ง 4 ปี (ปี 65, 66, 67, 68)
pivot_eval = df_agg.pivot_table(index='program_id', columns='year', values='min_pct').dropna()

# กำหนดตัวแปร Train (65, 66, 67) และ Test (68)
X_train = np.array([65, 66, 67]).reshape(-1, 1)
X_test = np.array([68]).reshape(-1, 1)

y_true = pivot_eval[68].values
y_pred_baseline = []
y_pred_lr = []

for idx, row in pivot_eval.iterrows():
    y_train = row[[65, 66, 67]].values
    
    # วิธีที่ 1: Baseline (ใช้ค่าเฉลี่ย 3 ปีย้อนหลัง)
    y_pred_baseline.append(np.mean(y_train))
    
    # วิธีที่ 2: Linear Regression
    model = LinearRegression().fit(X_train, y_train)
    y_pred_lr.append(model.predict(X_test)[0])

# คำนวณความคลาดเคลื่อน (ยิ่งน้อยยิ่งดี)
mae_baseline = mean_absolute_error(y_true, y_pred_baseline)
mae_lr = mean_absolute_error(y_true, y_pred_lr)

print(f"Error จาก Baseline (ใช้ค่าเฉลี่ย): คลาดเคลื่อน {mae_baseline:.2f} %")
print(f"Error จาก Linear Regression     : คลาดเคลื่อน {mae_lr:.2f} %")


print("\n" + "="*50)
print(" 🚨 10 อันดับคณะที่ทำนายพลาดมากที่สุด (Error Analysis Worst 10)")
print("="*50)

# สร้างตารางดูผลลัพธ์
results = pd.DataFrame({
    'program_id': pivot_eval.index,
    'Actual_68': y_true,
    'Predicted_68': y_pred_lr
})

# หาความแตกต่างระหว่างคะแนนจริง กับที่โมเดลทาย
results['Error'] = abs(results['Actual_68'] - results['Predicted_68'])

# ดึง 10 อันดับที่ Error สูงสุด
worst_10 = results.nlargest(10, 'Error')

# นำไปเชื่อมกับชื่อมหาลัยและคณะเพื่อให้ดูรู้เรื่อง
worst_10_detail = worst_10.merge(
    df_agg[['program_id', 'institution', 'faculty']].drop_duplicates(), 
    on='program_id', 
    how='left'
)

# จัดฟอร์แมตตัวเลขให้ดูง่ายขึ้น
worst_10_detail['Actual_68'] = worst_10_detail['Actual_68'].round(2)
worst_10_detail['Predicted_68'] = worst_10_detail['Predicted_68'].round(2)
worst_10_detail['Error'] = worst_10_detail['Error'].round(2)

print(worst_10_detail[['institution', 'faculty', 'Actual_68', 'Predicted_68', 'Error']].to_string(index=False))

# =================================================================
# ส่วนที่เพิ่มใหม่: พล็อตกราฟหลักฐานความผิดพลาด (Worst Cases Evidence)
# =================================================================
print("\n🎨 กำลังสร้างกราฟหลักฐานความผิดพลาด (eda_worst_cases_evidence.png)...")

# ดึงรหัสคณะที่เป็นตัวแทนของ Overshooting และ Undershooting ออกมา
try:
    # 1. เคสทายทะลุร้อย (Overshooting): วิทยาลัยนครราชสีมา คณะเทคนิคการแพทย์
    pid_over = worst_10_detail[worst_10_detail['faculty'].astype(str).str.contains('เทคนิคการแพทย์', na=False)]['program_id'].iloc[0]
    # 2. เคสทายติดลบ (Undershooting): มทร.รัตนโกสินทร์ คณะวิศวกรรมศาสตร์ 
    pid_under = worst_10_detail[worst_10_detail['institution'].astype(str).str.contains('รัตนโกสินทร์', na=False) & 
                               worst_10_detail['faculty'].astype(str).str.contains('วิศวกรรมศาสตร์', na=False)]['program_id'].iloc[0]

    # ตั้งค่ากราฟ 1 แถว 2 คอลัมน์
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cases = [(pid_over, 'Overshooting (ทายทะลุเพดาน)'), (pid_under, 'Undershooting (ทายติดลบ)')]

    for i, (pid, title_type) in enumerate(cases):
        ax = axes[i]
        row_data = pivot_eval.loc[pid]
        
        years_train = [65, 66, 67]
        scores_train = [row_data[65], row_data[66], row_data[67]]
        
        # จำลองเส้นตรง Linear Regression ของคณะนั้นๆ เหมือนที่โมเดลคำนวณจริง
        reg = LinearRegression().fit(np.array(years_train).reshape(-1,1), np.array(scores_train))
        all_years = [65, 66, 67, 68]
        line_preds = reg.predict(np.array(all_years).reshape(-1,1))
        
        # 1. พล็อตจุดข้อมูลจริงปี 65-67 (จุดสีฟ้า)
        ax.scatter(years_train, scores_train, color='#3498db', s=120, label='คะแนนจริง (65-67)', zorder=5)
        # 2. พล็อตเส้นตรงแนวโน้มยาวไปจนถึงปี 68 (เส้นประสีแดง)
        ax.plot(all_years, line_preds, color='#e74c3c', linestyle='--', linewidth=2, label='เส้นแนวโน้มโมเดล (Linear Trend)')
        # 3. พล็อตจุดที่โมเดลทายผิดพลาดในปี 68 (กากบาทสีแดง)
        ax.scatter([68], [line_preds[-1]], color='#e74c3c', marker='x', s=180, linewidths=3, label=f'โมเดลทำนายปี 68 ({line_preds[-1]:.1f}%)', zorder=6)
        # 4. พล็อตจุดคะแนนจริงที่เกิดขึ้นในปี 68 (จุดสีเขียว)
        ax.scatter([68], [row_data[68]], color='#2ecc71', marker='o', s=120, label=f'คะแนนจริงปี 68 ({row_data[68]:.1f}%)', zorder=5)
        
        # ดึงชื่อมหาลัย/คณะมาแสดงบนหัวกราฟ
        uni_name = worst_10_detail[worst_10_detail['program_id']==pid]['institution'].iloc[0]
        fac_name = worst_10_detail[worst_10_detail['program_id']==pid]['faculty'].iloc[0]
        
        ax.set_title(f"{uni_name}\n{fac_name}\n⚠️ {title_type}", fontsize=12, fontweight='bold')
        ax.set_xlabel('ปี TCAS', fontsize=10)
        ax.set_ylabel('คะแนนต่ำสุด (%)', fontsize=10)
        ax.set_xticks([65, 66, 67, 68])
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig('eda_worst_cases_evidence.png', dpi=300)
    print("✅ เซฟรูปหลักฐานเรียบร้อยในชื่อไฟล์: eda_worst_cases_evidence.png")
    plt.show()

except Exception as e:
    print(f"❌ ไม่สามารถสร้างกราฟหลักฐานได้เนื่องจาก: {e}")