// Browser UI bridge. Image analysis is performed by scanner.py through Flask.
const $ = id => document.getElementById(id);
const setText = (id, value) => { const element = $(id); if (element) element.textContent = value; };
let hasScan = false;
let latestResult = null;

function recommend(blank) {
  const sizes = { scratch: '4 x 6 in', memo: '3 x 5 in', flashcards: '3 x 5 in', sticky: '3 x 3 in', mini_sticky: '1.5 x 2 in' };
  const labels = { scratch: 'scratch sheets', memo: 'memo sheets', flashcards: 'flashcards', sticky: 'sticky notes', mini_sticky: 'mini sticky notes' };
  const inputs = [...document.querySelectorAll('.demand-card input[type=range][data-demand]')];
  const values = inputs.length ? inputs.map(input => ({ name: input.dataset.demand, value: Number(input.value), size: sizes[input.dataset.demand] })) : [{ name: 'sticky', value: 90, size: sizes.sticky }];
  const demand = values.sort((a, b) => b.value - a.value)[0];
  const enough = blank >= 20;
  setText('decision', enough ? 'Reuse recommended' : 'Recycle recommended'); setText('recommendation', enough ? `Cut into ${labels[demand.name]}` : 'Send to recycling');
  setText('recommendationCopy', enough ? `Best match for today's highest demand: ${labels[demand.name]} · typical size ${demand.size}.` : 'There is not enough usable area to justify cutting waste.');
  setText('pieceCount', enough ? Math.max(1, Math.floor(blank / 28)) : 0); setText('recovered', enough ? `${blank}%` : '0%'); setText('waste', enough ? `${Math.max(4, 100 - blank - 20)}%` : '0%');
  setText('processed', '129'); setText('reused', enough ? '87' : '86'); setText('diverted', enough ? '68%' : '67%');
}

function showResult(result) {
  latestResult = result; hasScan = true; setText('blankStat', `${result.blank}%`); setText('usableLegend', `${result.blank}%`); setText('printedStat', `${result.unusable ?? (100 - result.blank).toFixed(1)}%`); setText('damageStat', `${result.damage}%`); setText('confidence', `${result.confidence}% confidence · ${result.type}`);
  const meter = $('meterFill'); if (meter) meter.style.width = `${result.blank}%`;
  const planImage = document.querySelector('.cutting-plan-image');
  if (planImage && result.plan_image) { planImage.src = result.plan_image; planImage.alt = result.plan_status || 'OpenCV-generated cutting plan'; planImage.classList.remove('missing'); }
  const counts = { sticky: 0, mini_sticky: 0, memo: 0, scratch: 0, flashcards: 0 };
  (result.pieces || []).forEach(piece => { if (Object.prototype.hasOwnProperty.call(counts, piece.product)) counts[piece.product] += 1; });
  setText('countSticky', counts.sticky); setText('countMiniSticky', counts.mini_sticky); setText('countMemo', counts.memo);
  setText('countScratch', counts.scratch); setText('countFlashcards', counts.flashcards);
  const gcodeButton = $('downloadGcode'); if (gcodeButton) gcodeButton.disabled = !(result.pieces && result.pieces.length);
  const executeButton = $('executeGcode'); if (executeButton) executeButton.disabled = !(result.pieces && result.pieces.length);
  recommend(result.blank);
}

async function downloadGcode() {
  if (!latestResult || !latestResult.pieces?.length) return;
  const response = await fetch('/api/cutting/gcode', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pieces: latestResult.pieces, image_width: latestResult.width, image_height: latestResult.height, config: { paper_width_mm: latestResult.paper_width_mm, paper_height_mm: latestResult.paper_height_mm, max_x_mm: 310, max_y_mm: 300, pen_up_z_mm: 44, pen_down_z_mm: 41 } }) });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) return setText('confidence', `G-code failed: ${result.error || response.status}`);
  const blob = new Blob([result.gcode], { type: 'text/plain' });
  const link = document.createElement('a'); link.href = URL.createObjectURL(blob); link.download = 'repaper-cut-plan.gcode'; link.click(); URL.revokeObjectURL(link.href);
}

async function executeGcode() {
  if (!latestResult?.pieces?.length) return;
  const confirmed = window.confirm('This will home the printer, raise the blade to Z44, and execute the blade-cutting program from the paper bottom-right origin. Ensure the paper, blade, and work area are ready. Continue?');
  if (!confirmed) return;
  const log = $('machineLog'); if (log) log.textContent = 'Starting printer execution...';
  setText('confidence', 'Sending homing and cutting commands to the printer...');
  const response = await fetch('/api/cutting/execute', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pieces: latestResult.pieces, image_width: latestResult.width, image_height: latestResult.height, config: { paper_width_mm: latestResult.paper_width_mm, paper_height_mm: latestResult.paper_height_mm, max_x_mm: 310, max_y_mm: 300, pen_up_z_mm: 44, pen_down_z_mm: 41 } }) });
  const result = await response.json().catch(() => ({}));
  if (log) log.textContent = (result.logs || [`ERROR: ${result.error || response.status}`]).join('\\n');
  setText('confidence', response.ok ? `Printer completed ${result.sent_commands} commands.` : `Printer failed: ${result.error || response.status}`);
}

async function acceptFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) return setText('confidence', 'Please choose an image file');
  if (file.size > 10 * 1024 * 1024) return setText('confidence', 'Please choose an image under 10 MB');
  const source = URL.createObjectURL(file); const preview = $('preview');
  if (preview) { preview.src = source; preview.hidden = false; }
  const content = $('dropContent'); if (content) content.hidden = true;
  const analysis = $('analysis'); if (analysis) analysis.classList.remove('hidden'); setText('confidence', 'Analyzing with Python...');
  const body = new FormData(); body.append('image', file, file.name || 'scan.jpg'); body.append('demand', JSON.stringify(Object.fromEntries([...document.querySelectorAll('.demand-card input[type=range][data-demand]')].map(input => [input.dataset.demand, Number(input.value)]))));
  const uploadButton = $('uploadButton');
  if (uploadButton) uploadButton.disabled = true;
  try {
    const apiUrl = location.port === '5000' ? '/api/analyze' : 'http://127.0.0.1:5000/api/analyze';
    const response = await fetch(apiUrl, { method: 'POST', body }); const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `Scanner returned ${response.status}`); showResult(result);
  } catch (error) {
    setText('confidence', `Upload failed: ${error.message || 'Start scanner.py first'}`);
  } finally {
    if (uploadButton) uploadButton.disabled = false;
    URL.revokeObjectURL(source);
  }
}

let cameraStream = null;
function stopCamera() {
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  const panel = $('cameraPanel'); const video = $('cameraPreview');
  if (video) video.srcObject = null;
  if (panel) panel.hidden = true;
}
async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setText('confidence', 'Camera access requires a browser on localhost or HTTPS.');
    return;
  }
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    const video = $('cameraPreview'); const panel = $('cameraPanel');
    video.srcObject = cameraStream; panel.hidden = false; await video.play();
    setText('confidence', 'Camera ready — place the whole paper in view.');
  } catch (error) {
    setText('confidence', `Camera unavailable: ${error.message || 'permission denied'}`);
    stopCamera();
  }
}
async function automateScan() {
  const button = $('automateScanButton');
  if (button) button.disabled = true;
  setText('confidence', 'Homing printer and moving to camera position...');
  try {
    const response = await fetch('/api/automation/prepare-scan', { method: 'POST' });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || 'Printer preparation failed');
    const log = $('machineLog');
    if (log) log.textContent = (result.logs || []).join('\\n');
    await openCamera();
    setText('confidence', 'Camera ready. Capturing paper in 2 seconds...');
    window.setTimeout(captureCameraFrame, 2000);
  } catch (error) {
    setText('confidence', `Automation failed: ${error.message || error}`);
  } finally {
    if (button) button.disabled = false;
  }
}
function captureCameraFrame() {
  const video = $('cameraPreview'); const canvas = $('cameraCanvas');
  if (!video || !canvas || !video.videoWidth) return setText('confidence', 'Camera is not ready yet.');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(blob => { if (blob) { stopCamera(); acceptFile(new File([blob], 'usb-camera-scan.jpg', { type: 'image/jpeg' })); } }, 'image/jpeg', .92);
}

const fileInput = $('fileInput');
const uploadButton = $('uploadButton');
if (fileInput) fileInput.addEventListener('change', event => acceptFile(event.target.files[0]));
const dropzone = $('dropzone'); if (dropzone) { dropzone.addEventListener('dragover', event => { event.preventDefault(); dropzone.classList.add('dragging'); }); dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragging')); dropzone.addEventListener('drop', event => { event.preventDefault(); dropzone.classList.remove('dragging'); acceptFile(event.dataTransfer.files[0]); }); }
const sampleBtn = $('sampleBtn'); if (sampleBtn) sampleBtn.addEventListener('click', async () => { const response = await fetch('Assets/Sample_Worksheet.png'); const blob = await response.blob(); acceptFile(new File([blob], 'sample.png', { type: 'image/png' })); });
const cameraButton = $('cameraButton'); if (cameraButton) cameraButton.addEventListener('click', openCamera);
const automateScanButton = $('automateScanButton'); if (automateScanButton) automateScanButton.addEventListener('click', automateScan);
const captureButton = $('captureButton'); if (captureButton) captureButton.addEventListener('click', captureCameraFrame);
const closeCameraButton = $('closeCameraButton'); if (closeCameraButton) closeCameraButton.addEventListener('click', stopCamera);
function updateDemandSummary() {
  const labels = { sticky: 'Sticky Notes', mini_sticky: 'Mini Sticky Notes', scratch: 'Scratch Sheets', memo: 'Memo Sheets', flashcards: 'Flashcards' };
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
const demandKeys = ['sticky', 'mini_sticky', 'scratch', 'memo', 'flashcards'];
function dashboardDemandInput(key) { return document.querySelector(`.demand-card input[data-demand="${key}"]`); }
function settingsDemandInput(key) { return document.querySelector(`#modal input[type="range"][data-demand="${key}"]`); }
function syncSettingsFromDashboard() {
  demandKeys.forEach(key => {
    const source = dashboardDemandInput(key); const target = settingsDemandInput(key);
    if (!source || !target) return;
    target.value = source.value;
    const valueId = { sticky: 'stickyValue', mini_sticky: 'miniStickyValue', scratch: 'scratchValue', memo: 'memoValue', flashcards: 'flashValue' }[key];
    setText(valueId, `${source.value}%`);
  });
}
function syncDashboardFromSettings() {
  demandKeys.forEach(key => {
    const source = settingsDemandInput(key); const target = dashboardDemandInput(key);
    if (!source || !target) return;
    target.value = source.value;
    const output = target.nextElementSibling;
    if (output) { output.value = source.value; output.textContent = `${source.value}%`; }
  });
  updateDemandSummary();
  const blank = Number.parseInt(($('blankStat') || {}).textContent, 10);
  if (hasScan && Number.isFinite(blank)) recommend(blank);
}
function updateSliderPercentage(input) {
  const value = `${input.value}%`;
  const rowOutput = input.closest('.demand-row')?.querySelector('output');
  if (rowOutput) rowOutput.textContent = value;
  const valueIds = { sticky: 'stickyValue', mini_sticky: 'miniStickyValue', scratch: 'scratchValue', memo: 'memoValue', flashcards: 'flashValue' };
  if (valueIds[input.dataset.demand]) setText(valueIds[input.dataset.demand], value);
}
document.querySelectorAll('.demand-card input[type=range][data-demand]').forEach(input => input.addEventListener('input', event => { updateSliderPercentage(event.target); syncSettingsFromDashboard(); updateDemandSummary(); const blank = Number.parseInt(($('blankStat') || {}).textContent, 10); if (hasScan && Number.isFinite(blank)) recommend(blank); }));
document.querySelectorAll('#modal input[type=range][data-demand]').forEach(input => input.addEventListener('input', event => { updateSliderPercentage(event.target); syncDashboardFromSettings(); }));
syncSettingsFromDashboard();
updateDemandSummary();
const printBtn = $('printBtn'); if (printBtn) printBtn.addEventListener('click', () => window.print());
const gcodeButton = $('downloadGcode'); if (gcodeButton) gcodeButton.addEventListener('click', downloadGcode);
const executeButton = $('executeGcode'); if (executeButton) executeButton.addEventListener('click', executeGcode);
