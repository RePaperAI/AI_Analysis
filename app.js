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
