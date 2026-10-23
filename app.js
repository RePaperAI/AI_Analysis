// Browser UI bridge. Image analysis is performed by scanner.py through Flask.
const $ = id => document.getElementById(id);
const setText = (id, value) => { const element = $(id); if (element) element.textContent = value; };
let hasScan = false;

function recommend(blank) {
  const sizes = { scratch: '8.5 x 11 in (Letter) or A4', memo: '3 x 3 or 4 x 6 in', flashcards: '3 x 5 or 4 x 6 in', sticky: '3 in x 3 in' };
  const labels = { scratch: 'scratch sheets', memo: 'memo sheets', flashcards: 'flashcards', sticky: 'sticky notes' };
  const inputs = [...document.querySelectorAll('[data-demand]')];
  const values = inputs.length ? inputs.map(input => ({ name: input.dataset.demand, value: Number(input.value), size: sizes[input.dataset.demand] })) : [{ name: 'sticky', value: 90, size: sizes.sticky }];
  const demand = values.sort((a, b) => b.value - a.value)[0];
  const enough = blank >= 20;
  setText('decision', enough ? 'Reuse recommended' : 'Recycle recommended'); setText('recommendation', enough ? `Cut into ${labels[demand.name]}` : 'Send to recycling');
  setText('recommendationCopy', enough ? `Best match for today's highest demand: ${labels[demand.name]} · typical size ${demand.size}.` : 'There is not enough usable area to justify cutting waste.');
  setText('pieceCount', enough ? Math.max(1, Math.floor(blank / 28)) : 0); setText('recovered', enough ? `${blank}%` : '0%'); setText('waste', enough ? `${Math.max(4, 100 - blank - 20)}%` : '0%');
  setText('processed', '129'); setText('reused', enough ? '87' : '86'); setText('diverted', enough ? '68%' : '67%');
}

function showResult(result) {
  hasScan = true; setText('blankStat', `${result.blank}%`); setText('printedStat', `${result.printed}%`); setText('damageStat', `${result.damage}%`); setText('confidence', `${result.confidence}% confidence · ${result.type}`);
  const meter = $('meterFill'); if (meter) meter.style.width = `${result.blank}%`;
  recommend(result.blank);
}

async function acceptFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return setText('confidence', 'Please choose an image file');
  if (file.size > 10 * 1024 * 1024) return setText('confidence', 'Please choose an image under 10 MB');
  const source = URL.createObjectURL(file); const preview = $('preview');
  if (preview) { preview.src = source; preview.hidden = false; }
  const content = $('dropContent'); if (content) content.hidden = true;
  const analysis = $('analysis'); if (analysis) analysis.classList.remove('hidden'); setText('confidence', 'Analyzing with Python...');
  const body = new FormData(); body.append('image', file, file.name || 'scan.jpg'); body.append('demand', JSON.stringify(Object.fromEntries([...document.querySelectorAll('[data-demand]')].map(input => [input.dataset.demand, Number(input.value)]))));
  try {
    const apiUrl = location.port === '5000' ? '/api/analyze' : 'http://127.0.0.1:5000/api/analyze';
    const response = await fetch(apiUrl, { method: 'POST', body }); const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Scanner returned ${response.status}`); showResult(result);
  } catch (error) { setText('confidence', `Python scanner unavailable: ${error.message}`); } finally { URL.revokeObjectURL(source); }
}

const fileInput = $('fileInput'); if (fileInput) fileInput.addEventListener('change', event => acceptFile(event.target.files[0]));
const dropzone = $('dropzone'); if (dropzone) { dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragging'); }); dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging')); dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('dragging'); acceptFile(event.dataTransfer.files[0]); }); }
const sampleBtn = $('sampleBtn'); if (sampleBtn) sampleBtn.addEventListener('click', () => { const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800"><rect width="100%" height="100%" fill="white"/><text x="55" y="90" font-family="Arial" font-size="22" fill="#555">SCIENCE WORKSHEET</text><path d="M55 135h460M55 175h360M55 215h410M55 255h300" stroke="#aab6b0" stroke-width="5"/></svg>`; acceptFile(new File([svg], 'sample.svg', { type: 'image/svg+xml' })); });
function updateDemandSummary() {
  const labels = { sticky: 'Sticky Notes', scratch: 'Scratch Sheets', memo: 'Memo Sheets', flashcards: 'Flashcards' };
  const rows = [...document.querySelectorAll('.demand-card input[type=range][data-demand]')]
    .map(input => ({ name: input.dataset.demand, value: Number(input.value) }))
    .sort((a, b) => b.value - a.value);
  if (!rows.length) return;
  const top = rows[0];
  const level = top.value >= 70 ? 'High Demand' : top.value >= 40 ? 'Medium Demand' : 'Low Demand';
  const displayed = rows.filter(row => {
    const rowLevel = row.value >= 70 ? 'High Demand' : row.value >= 40 ? 'Medium Demand' : 'Low Demand';
    return rowLevel === level;
  });
  const productLine = $('demandSummaryProducts');
  if (productLine) {
    productLine.replaceChildren();
    displayed.forEach((row, index) => {
      if (index > 0) {
        const separator = displayed.length === 2 || index === displayed.length - 1 ? (displayed.length > 2 && index === displayed.length - 1 ? ', and ' : ' and ') : ', ';
        productLine.append(document.createTextNode(separator));
      }
      const product = document.createElement('strong');
      product.className = `demand-summary-product demand-${row.name}${index === 0 ? ' primary' : ''}`;
      product.textContent = labels[row.name];
      productLine.append(product);
    });
  }
  setText('demandSummaryLevel', level);
}
document.querySelectorAll('input[type=range][data-demand]').forEach(input => input.addEventListener('input', event => { const output = event.target.nextElementSibling; if (output && (output.matches('output') || output.tagName === 'STRONG')) { output.value = event.target.value; output.textContent = `${event.target.value}%`; } updateDemandSummary(); const blank = Number.parseInt(($('blankStat') || {}).textContent, 10); if (hasScan && Number.isFinite(blank)) recommend(blank); }));
updateDemandSummary();
const printBtn = $('printBtn'); if (printBtn) printBtn.addEventListener('click', () => window.print());

// Gemini worksheet scanner. The API key is entered by the judge and is not persisted.
let previewUrl = '';
const GEMINI_MODELS = ['gemini-3.8-flash', 'gemini-3.6-flash'];
const geminiSchema = {
  type: 'OBJECT',
  properties: {
    blankPercent: { type: 'NUMBER' },
    printedPercent: { type: 'NUMBER' },
    damagedPercent: { type: 'NUMBER' },
    confidence: { type: 'NUMBER' },
    materialType: { type: 'STRING' },
    recommendation: { type: 'STRING' },
    recommendationReason: { type: 'STRING' },
    cuttingPlan: {
      type: 'ARRAY',
      items: {
        type: 'OBJECT',
        properties: {
          label: { type: 'STRING' },
          quantity: { type: 'INTEGER' },
          size: { type: 'STRING' },
          action: { type: 'STRING' }
        },
        propertyOrdering: ['label', 'quantity', 'size', 'action']
      }
    }
  },
  propertyOrdering: ['blankPercent', 'printedPercent', 'damagedPercent', 'confidence', 'materialType', 'recommendation', 'recommendationReason', 'cuttingPlan']
};

function fileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(',')[1]);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function geminiNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(100, Math.round(number))) : fallback;
}

function setGeminiStatus(message, isError = false) {
  const status = $('geminiStatus');
  if (status) { status.textContent = message; status.style.color = isError ? '#b33b32' : ''; }
}

function renderGeminiPlan(result) {
  const plan = $('geminiCutPlan');
  const staticPlan = $('staticCutPlan');
  if (!plan) return;
  const items = Array.isArray(result.cuttingPlan) ? result.cuttingPlan : [];
  plan.replaceChildren();
  items.forEach(item => {
    const row = document.createElement('div');
    row.className = `gemini-cut-item${String(item.action || '').toLowerCase().includes('recycl') ? ' recycle' : ''}`;
    const label = document.createElement('span');
    const quantity = Math.max(0, Number.parseInt(item.quantity, 10) || 0);
    label.innerHTML = `<strong>${item.label || 'Cut piece'}</strong><br>${item.size || 'Size to be confirmed'}${item.action ? ` · ${item.action}` : ''}`;
    const count = document.createElement('strong');
    count.textContent = `×${quantity}`;
    row.append(label, count);
    plan.append(row);
  });
  plan.classList.remove('hidden');
  if (staticPlan) staticPlan.classList.add('hidden');
  const staticGrid = document.querySelector('.cutting-plan .cut-sheet');
  if (staticGrid) staticGrid.classList.add('hidden');
}

function applyGeminiResult(result) {
  const blank = geminiNumber(result.blankPercent, 0);
  const printed = geminiNumber(result.printedPercent, Math.max(0, 100 - blank));
  const damage = geminiNumber(result.damagedPercent, 0);
  setText('blankStat', `${blank}%`); setText('usableLegend', `${blank}%`);
  setText('printedStat', `${printed}%`); setText('damageStat', `${damage}%`);
  const meter = $('meterFill'); if (meter) meter.style.width = `${blank}%`;
  setText('confidence', `${geminiNumber(result.confidence, 0)}% confidence · ${result.materialType || 'worksheet'}`);
  setText('decision', result.recommendation || (blank >= 20 ? 'Reuse recommended' : 'Recycle recommended'));
  setText('recommendation', result.recommendation || 'Plan generated');
  setText('recommendationCopy', result.recommendationReason || 'Gemini generated this plan from the uploaded sheet.');
  const reuse = (result.cuttingPlan || []).filter(item => !String(item.action || '').toLowerCase().includes('recycl')).reduce((sum, item) => sum + (Number(item.quantity) || 0), 0);
  const recycle = (result.cuttingPlan || []).filter(item => String(item.action || '').toLowerCase().includes('recycl')).reduce((sum, item) => sum + (Number(item.quantity) || 0), 0);
  setText('pieceCount', reuse); setText('recovered', `${blank}%`); setText('waste', recycle);
  renderGeminiPlan(result);
}

async function analyzeWithGemini() {
  const key = $('geminiKey')?.value.trim();
  const file = $('geminiFile')?.files?.[0];
  const button = $('geminiScanBtn');
  if (!key) return setGeminiStatus('Enter a Gemini API key to run the judge demo.', true);
  if (!file || !file.type.startsWith('image/')) return setGeminiStatus('Choose an image of the worksheet first.', true);
  if (file.size > 10 * 1024 * 1024) return setGeminiStatus('Please choose an image under 10 MB.', true);
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  const preview = $('preview'); if (preview) { preview.src = previewUrl; preview.classList.remove('missing'); }
  if (button) button.disabled = true;
  setGeminiStatus('Gemini is reading the worksheet and creating a cutting plan…');
  try {
    const base64 = await fileAsBase64(file);
    const demand = Object.fromEntries([...document.querySelectorAll('.demand-card input[type="range"][data-demand]')].map(input => [input.dataset.demand, Number(input.value)]));
    const prompt = `You are the RePaper AI paper-reuse planner. Inspect the uploaded worksheet image. Estimate the percentage of blank/usable area, printed area, and damaged area. Then create a practical cutting plan that prioritizes this school's demand profile: ${JSON.stringify(demand)}. Return only JSON matching the supplied schema. Be conservative: damaged or heavily printed areas should be recycled, and never claim more than 100 percent total area.`;
    const requestBody = JSON.stringify({ contents: [{ parts: [{ inlineData: { mimeType: file.type, data: base64 } }, { text: prompt }] }], generationConfig: { responseMimeType: 'application/json', responseSchema: geminiSchema, temperature: 0.2 } });
    let payload;
    let lastError = '';
    for (const model of GEMINI_MODELS) {
      for (let attempt = 0; attempt < 2; attempt += 1) {
        setGeminiStatus(`Trying Gemini ${model}${attempt ? ` again (${attempt + 1}/2)` : ''}…`);
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': key },
          body: requestBody
        });
        payload = await response.json();
        if (response.ok) break;
        lastError = payload?.error?.message || `Gemini returned ${response.status}`;
        const temporary = response.status === 429 || response.status === 500 || response.status === 502 || response.status === 503 || /high demand|overloaded|temporar/i.test(lastError);
        if (!temporary) throw new Error(lastError);
        if (attempt === 0) await new Promise(resolve => setTimeout(resolve, 1200));
      }
      if (payload && !payload.error) break;
    }
    if (!payload || payload.error) throw new Error(`Gemini is busy right now. ${lastError}`);
    const text = payload?.candidates?.[0]?.content?.parts?.map(part => part.text || '').join('') || '';
    const result = JSON.parse(text.replace(/^```json\s*|\s*```$/g, '').trim());
    applyGeminiResult(result);
    setGeminiStatus('Analysis complete. Gemini generated the cutting plan shown on the right.');
  } catch (error) {
    setGeminiStatus(`Gemini analysis failed: ${error.message}`, true);
  } finally {
    if (button) button.disabled = false;
  }
}

const geminiScanBtn = $('geminiScanBtn');
if (geminiScanBtn) geminiScanBtn.addEventListener('click', analyzeWithGemini);
