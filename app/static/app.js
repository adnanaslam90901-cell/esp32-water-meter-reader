let latestId = null;
const $ = (id) => document.getElementById(id);
const endpoint = `${location.origin}/api/read?meter_id=meter-01`;
$('endpoint').textContent = endpoint;

function copyEndpoint(){ navigator.clipboard.writeText(endpoint); }
function when(ts){ return new Date(ts * 1000).toLocaleString(); }
function esc(v){ return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function scoreClass(score){ return score >= 75 ? 'good' : score >= 45 ? 'warn' : 'bad'; }

function showLatest(row){
  latestId = row.id;
  $('latestTitle').textContent = row.image_name || 'Latest image';
  $('latestTime').textContent = when(row.created_at || Math.floor(Date.now()/1000));
  $('photoBox').className = 'photo';
  $('photoBox').innerHTML = `<img src="${row.image_url}?v=${Date.now()}" alt="Latest water meter capture">`;
  $('readingInput').disabled = false; $('readingInput').value = row.reading || '';
  $('saveCorrection').disabled = false;
  $('meterId').textContent = row.meter_id || '—'; $('roiMode').textContent = row.roi_source || row.roi?.source || '—';
  const c = Number(row.confidence || 0); $('confidence').textContent = `${c.toFixed(1)}% confidence`; $('confidence').className = `badge ${scoreClass(c)}`;
}

async function refresh(){
  try{
    const rows = await fetch('/api/readings?limit=100').then(r=>r.json());
    $('serverState').className='state'; $('serverState').innerHTML='<i></i> Server online';
    if(rows.length && rows[0].id !== latestId) showLatest(rows[0]);
    $('history').innerHTML = rows.length ? rows.map(r=>`<tr><td><a href="${r.image_url}" target="_blank"><img src="${r.image_url}" alt="capture"></a></td><td>${esc(r.meter_id)}</td><td><b class="digits">${esc(r.reading) || 'Review needed'}</b>${r.corrected ? '<em>corrected</em>':''}</td><td><span class="dot ${scoreClass(r.confidence)}"></span>${Number(r.confidence).toFixed(1)}%</td><td>${when(r.created_at)}</td></tr>`).join('') : '<tr><td colspan="5" class="emptyrow">No readings yet.</td></tr>';
  }catch{ $('serverState').className='state off'; $('serverState').textContent='Server disconnected'; }
}

$('uploadForm').addEventListener('submit', async(e)=>{
  e.preventDefault(); const button=e.submitter; button.disabled=true; button.textContent='Reading…';
  const form=new FormData(); form.append('image',$('file').files[0]);
  try{ const response=await fetch(`/api/upload?meter_id=${encodeURIComponent($('manualMeter').value)}`,{method:'POST',body:form}); const row=await response.json(); if(!response.ok) throw new Error(row.detail); row.image_name=$('file').files[0].name; row.created_at=Math.floor(Date.now()/1000); showLatest(row); await refresh(); }
  catch(err){ $('message').textContent=err.message; } finally{ button.disabled=false; button.textContent='Read image'; }
});

$('saveCorrection').addEventListener('click',async()=>{
  const response=await fetch(`/api/readings/${latestId}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({reading:$('readingInput').value})});
  const data=await response.json(); $('message').textContent=response.ok?'Correction saved.':data.detail; if(response.ok) refresh();
});

$('autoRoi').addEventListener('change',()=>document.querySelector('.roi-fields').classList.toggle('disabled',$('autoRoi').checked));
$('saveConfig').addEventListener('click',async()=>{
  const roi=$('autoRoi').checked?null:{x:+$('roiX').value/100,y:+$('roiY').value/100,w:+$('roiW').value/100,h:+$('roiH').value/100};
  await fetch('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({roi,decimal_places:+$('decimalPlaces').value})});
  $('saveConfig').textContent='Saved'; setTimeout(()=>$('saveConfig').textContent='Save calibration',1200);
});

fetch('/api/config').then(r=>r.json()).then(c=>{ if(c.roi){$('autoRoi').checked=false; $('roiX').value=c.roi.x*100; $('roiY').value=c.roi.y*100; $('roiW').value=c.roi.w*100; $('roiH').value=c.roi.h*100;} $('decimalPlaces').value=c.decimal_places||0; document.querySelector('.roi-fields').classList.toggle('disabled',$('autoRoi').checked); });
// ---------------------------------------------------------------- dataset batches
let batchFiles = [], activeBatch = null, batchTimer = null;
const skippedDirs = /(^|\/)(masks?|collages?|__macosx)\//i;   // Kaggle masks/collages are not meter photos
const batchRunning = (status) => status === 'queued' || status === 'running';
const stateClass = { done:'good', running:'warn', queued:'warn', cancelled:'bad', interrupted:'bad', failed:'bad' };

function pickBatchFiles(list){
  batchFiles = [...list].filter(f => { const p = f.webkitRelativePath || f.name; return !skippedDirs.test(p) && /\.(jpe?g|png|bmp|webp|zip|csv)$/i.test(p); });
  const zips = batchFiles.filter(f => /\.zip$/i.test(f.name)).length, csvs = batchFiles.filter(f => /\.csv$/i.test(f.name)).length;
  const photos = batchFiles.length - zips - csvs;
  $('batchPicked').textContent = batchFiles.length
    ? [photos && `${photos.toLocaleString()} photos`, zips && `${zips} ZIP`, csvs && `${csvs} labels CSV`].filter(Boolean).join(' · ')
    : 'No photos, ZIP or CSV in that selection';
  $('batchStart').disabled = !batchFiles.length;
}
$('batchPick').addEventListener('change', e => pickBatchFiles(e.target.files));
$('batchFolder').addEventListener('change', e => pickBatchFiles(e.target.files));

$('batchForm').addEventListener('submit', e => {
  e.preventDefault();
  const form = new FormData();
  batchFiles.forEach(f => form.append('files', f, f.webkitRelativePath || f.name));
  const xhr = new XMLHttpRequest(), button = $('batchStart');
  xhr.open('POST', `/api/batches?meter_id=${encodeURIComponent($('batchMeter').value)}`);
  button.disabled = true; button.textContent = 'Uploading…';
  xhr.upload.onprogress = ev => { if (ev.lengthComputable) button.textContent = `Uploading ${Math.round(ev.loaded / ev.total * 100)}%`; };
  const done = (message) => { button.disabled = !batchFiles.length; button.textContent = 'Start reading'; if (message) $('batchPicked').textContent = message; };
  xhr.onerror = () => done('Upload failed. Is the server running?');
  xhr.onload = () => {
    let data = {}; try { data = JSON.parse(xhr.responseText); } catch {}
    if (xhr.status >= 400) return done(data.detail || 'Upload failed');
    done(`Batch ${data.id} started: ${data.total.toLocaleString()} photos` + (data.labelled ? `, ${data.labelled.toLocaleString()} with expected values` : ''));
    openBatch(data.id); loadBatches();
  };
  xhr.send(form);
});

function openBatch(id){
  activeBatch = id; $('batchPanel').hidden = false;
  $('batchXlsx').href = `/api/export.xlsx?batch_id=${id}`; $('batchCsv').href = `/api/export.csv?batch_id=${id}`;
  pollBatch();
}

async function pollBatch(){
  clearTimeout(batchTimer);
  const id = activeBatch; if (!id) return;
  let batch, rows;
  try{
    const response = await fetch(`/api/batches/${id}`);
    if (!response.ok){ $('batchPanel').hidden = true; activeBatch = null; return; }
    batch = await response.json();
    rows = await fetch(`/api/batches/${id}/readings?show=${$('batchShow').value}&limit=200`).then(r => r.json());
  }catch{ batchTimer = setTimeout(pollBatch, 4000); return; }
  if (id !== activeBatch) return;
  $('batchNo').textContent = batch.id; $('batchName').textContent = batch.name;
  $('batchState').textContent = batch.status; $('batchState').className = `badge ${stateClass[batch.status] || ''}`;
  $('batchCancel').hidden = !batchRunning(batch.status);
  $('batchBar').style.width = `${batch.total ? batch.processed / batch.total * 100 : 0}%`;
  $('batchSummary').innerHTML = [{label:'Read so far', value:`${batch.processed.toLocaleString()} / ${batch.total.toLocaleString()}`}, ...batch.summary.filter(s => s.label !== 'Images')]
    .map(s => `<div><span>${esc(s.label)}</span><b>${esc(s.value)}</b></div>`).join('');
  $('batchCount').textContent = rows.length === 200 ? 'Showing the latest 200 · download for everything' : `${rows.length.toLocaleString()} shown`;
  $('batchRows').innerHTML = rows.length ? rows.map(r => `<tr>
      <td>${r.image_url ? `<a href="${r.image_url}" target="_blank"><img src="${r.image_url}" alt="capture" loading="lazy"></a>` : ''}</td>
      <td class="file" title="${esc(r.source_name)}">${esc(r.source_name)}${r.note ? `<em class="err">${esc(r.note)}</em>` : ''}</td>
      <td><b class="digits">${esc(r.reading) || '—'}</b></td>
      <td class="digits">${esc(r.expected) || '—'}</td>
      <td>${r.match ? `<span class="badge ${r.match === 'Yes' ? 'good' : 'bad'}">${r.match}</span>` : '—'}</td>
      <td><span class="dot ${scoreClass(r.confidence)}"></span>${Number(r.confidence).toFixed(1)}%</td></tr>`).join('')
    : '<tr><td colspan="6" class="emptyrow">Nothing here yet.</td></tr>';
  if (batchRunning(batch.status)) batchTimer = setTimeout(pollBatch, 2000); else loadBatches();
}

async function loadBatches(){
  const batches = await fetch('/api/batches').then(r => r.json()).catch(() => null);
  if (!batches) return;
  $('batchList').innerHTML = batches.length ? batches.map(b => `<tr>
      <td><b>#${b.id}</b> ${esc(b.name)}</td><td>${b.processed.toLocaleString()} / ${b.total.toLocaleString()}</td>
      <td>${b.accuracy == null ? '—' : `${b.accuracy}%`}</td><td><span class="badge ${stateClass[b.status] || ''}">${b.status}</span></td><td>${when(b.created_at)}</td>
      <td class="rowactions"><button data-open="${b.id}">View</button><a href="/api/export.xlsx?batch_id=${b.id}">Excel</a><a href="/api/export.csv?batch_id=${b.id}">CSV</a>${batchRunning(b.status) ? '' : `<button data-delete="${b.id}" class="danger">Delete</button>`}</td></tr>`).join('')
    : '<tr><td colspan="6" class="emptyrow">No batches yet.</td></tr>';
  return batches;
}

$('batchList').addEventListener('click', async e => {
  const open = e.target.dataset.open, del = e.target.dataset.delete;
  if (open) openBatch(Number(open));
  if (del && confirm(`Delete batch ${del} and its stored previews?`)){
    await fetch(`/api/batches/${del}`, { method:'DELETE' });
    if (Number(del) === activeBatch){ activeBatch = null; $('batchPanel').hidden = true; }
    loadBatches();
  }
});
$('batchCancel').addEventListener('click', async () => { if (activeBatch){ await fetch(`/api/batches/${activeBatch}/cancel`, { method:'POST' }); pollBatch(); } });
$('batchShow').addEventListener('change', pollBatch);
loadBatches().then(batches => { const show = batches?.find(b => batchRunning(b.status)) || batches?.[0]; if (show) openBatch(show.id); });

refresh(); setInterval(refresh,4000);
