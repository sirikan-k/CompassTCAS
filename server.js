const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const xlsx = require('xlsx');

// แยกแต่ละบรรทัด CSV อย่างถูกต้อง รองรับ field ที่ครอบด้วย " " และมี , หรือ (...) อยู่ข้างใน
function parseCSVLine(line) {
    const result = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
            if (inQuotes && line[i+1] === '"') { cur += '"'; i++; }
            else { inQuotes = !inQuotes; }
        } else if (ch === ',' && !inQuotes) {
            result.push(cur);
            cur = '';
        } else {
            cur += ch;
        }
    }
    result.push(cur);
    return result.map(v => v.trim());
}

// แยกไฟล์ CSV ทั้งไฟล์ออกเป็น "บรรทัดข้อมูล" โดยรองรับ \n ที่อยู่ภายใน quoted field
function splitCSVRecords(content) {
    const records = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < content.length; i++) {
        const ch = content[i];
        if (ch === '"') inQuotes = !inQuotes;
        if ((ch === '\n') && !inQuotes) {
            records.push(cur);
            cur = '';
        } else if (ch !== '\r') {
            cur += ch;
        }
    }
    if (cur.trim()) records.push(cur);
    return records;
}

const app = express();
app.use(cors());
app.use(express.json());

let tcasDatabase = [];

// อ่านไฟล์ Excel แล้วดึงคอลัมน์ที่ต้องการออกมา
function readExcelMinScore(filename, programIdCol, minScoreCol) {
    const filePath = path.join(__dirname, 'data', filename);
    if (!fs.existsSync(filePath)) {
        console.warn(`⚠️ ไม่พบไฟล์ ${filename}`);
        return {};
    }
    const wb = xlsx.readFile(filePath);
    const ws = wb.Sheets[wb.SheetNames[0]];
    const rows = xlsx.utils.sheet_to_json(ws);

    const result = {};
    rows.forEach(row => {
        const id = String(row[programIdCol] || '').trim();
        const score = parseFloat(row[minScoreCol]);
        if (id && !isNaN(score) && score > 0) {
            result[id] = score;
        }
    });
    return result;
}

function loadDatabase() {
    // อ่านคะแนนต่ำสุดจากไฟล์ Excel แต่ละปี
    const min65 = readExcelMinScore('TCAS65_maxmin.xlsx', 'program_id', 'คะแนนต่ำสุด หลังประมวลผลรอบ 2');
    const min66 = readExcelMinScore('TCAS66_maxmin.xlsx', 'รหัสหลักสูตร', 'คะแนนต่ำสุด');
    const min67 = readExcelMinScore('TCAS67_maxmin.xlsx', 'รหัสหลักสูตร', 'คะแนนต่ำสุด หลังประมวลผลรอบ 2');
    const min68 = readExcelMinScore('TCAS68_maxmin.xlsx', 'รหัสหลักสูตร', 'คะแนนต่ำสุด ประมวลผลครั้งที่ 2');

    console.log(`📊 โหลดคะแนนปี 65: ${Object.keys(min65).length} รายการ`);
    console.log(`📊 โหลดคะแนนปี 66: ${Object.keys(min66).length} รายการ`);
    console.log(`📊 โหลดคะแนนปี 67: ${Object.keys(min67).length} รายการ`);
    console.log(`📊 โหลดคะแนนปี 68: ${Object.keys(min68).length} รายการ`);

    // อ่าน CSV หลัก
    const csvPath = path.join(__dirname, 'data', 'tcas_round3_full_data.csv');
    const altPath = path.join(__dirname, 'tcas_round3_full_data.csv');
    const finalPath = fs.existsSync(csvPath) ? csvPath : altPath;

    if (!fs.existsSync(finalPath)) {
        console.error("❌ ไม่พบไฟล์ tcas_round3_full_data.csv");
        return;
    }

    const content = fs.readFileSync(finalPath, 'utf-8');
    const lines = splitCSVRecords(content);
    const headers = parseCSVLine(lines[0]).map(h => h.trim().replace(/^"|"$/g, ''));

    for (let i = 1; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const row = parseCSVLine(lines[i]).map(v => v.trim().replace(/^"|"$/g, ''));

        let item = {};
        headers.forEach((header, index) => { item[header] = row[index] || ''; });

        if (item.program_id) {
            let criteria = {};
            try {
                criteria = JSON.parse((item.scores_criteria || '{}').replace(/'/g, '"'));
            } catch (e) { criteria = {}; }

            const id = item.program_id.trim();

            tcasDatabase.push({
                id,
                uni: item.university_name,
                group: item.faculty_name,
                program: item.program_name,
                gpax_min: parseFloat(item.min_gpax) || 0,
                // คะแนนต่ำสุดแต่ละปี ดึงจาก Excel จริงๆ
                min65: min65[id] || null,
                min66: min66[id] || null,
                min67: min67[id] || null,
                min68: min68[id] || null,
                criteria,
                link: item.link || '#'
            });
        }
    }
    console.log(`🟢 โหลดฐานข้อมูลเสร็จ! ทั้งหมด ${tcasDatabase.length} หลักสูตร`);
}

app.get('/api/database', (req, res) => res.json(tcasDatabase));

app.get('/api/search', (req, res) => {
    const query = (req.query.q || '').toLowerCase();
    if (query.length < 2) return res.json([]);
    const results = tcasDatabase.filter(item =>
        item.uni.toLowerCase().includes(query) ||
        item.program.toLowerCase().includes(query) ||
        item.group.toLowerCase().includes(query)
    ).slice(0, 50);
    res.json(results);
});

app.get('/api/groups', (req, res) => {
    res.json([...new Set(tcasDatabase.map(item => item.group))]);
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`🚀 Server รันแล้วที่ http://localhost:${PORT}`);
    loadDatabase();
});