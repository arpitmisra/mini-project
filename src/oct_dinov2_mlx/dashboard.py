"""Self-contained offline HTML viewer. No web server, uploads or CDN dependencies."""
import base64
import io
import json

import numpy as np
from PIL import Image


def render_dashboard(path, arrays, report):
    buffer = io.BytesIO()
    Image.fromarray(arrays["crop_rgb"]).save(buffer, format="PNG")
    payload = {"image": base64.b64encode(buffer.getvalue()).decode(), "report": report, "arrays": {}}
    for mode in ("imagenet", "unit"):
        for key in ("attention", "cosine", "pearson", "valid_cosine", "valid_pearson"):
            name = mode + "_" + key
            payload["arrays"][name] = base64.b64encode(np.asarray(arrays[name], dtype="<f4").tobytes()).decode()
    # Escape HTML delimiters even inside JSON strings (untrusted filenames).
    data = json.dumps(payload).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    path.write_text(TEMPLATE.replace("__PAYLOAD__", data), encoding="utf-8")


TEMPLATE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCT patch explorer</title>
<style>
body{font:16px system-ui;background:#f1f5f9;color:#14243a;margin:0;padding:28px;max-width:1300px;margin:auto}
h1{margin:0}p{line-height:1.5}small{color:#475569}label{display:inline-block;margin:8px 20px 8px 0}
select,input{font:inherit;padding:6px;max-width:220px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px}
section{background:white;border:1px solid #dbe4ee;border-radius:14px;padding:18px}canvas{width:100%;max-width:384px;image-rendering:pixelated;cursor:crosshair}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px} .note{padding:12px;border-left:4px solid #2563eb;background:#e7f0ff}
</style>
<h1>OCT patch explorer</h1><p id="subtitle"></p>
<p class="note">Click a patch in the image or a cell in the all-pairs matrix. Attention is directional; cosine and Pearson compare final patch descriptors. These views do not identify lesion boundaries.</p>
<label>Input preprocessing <select id="normalization"><option value="imagenet">Pretrained mean/std</option><option value="unit">RGB / 255 only</option></select></label>
<label>Relationship <select id="relation"><option value="attention">Attention weights</option><option value="cosine">Feature cosine similarity</option><option value="pearson">Feature Pearson correlation</option></select></label>
<label>Attention head <select id="head"><option value="mean">Mean of 6 heads</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<label>Query patch (0–255) <input id="query" type="number" min="0" max="255" value="123"></label>
<div class="grid">
<section><h2>Preprocessed crop</h2><canvas id="source" width="224" height="224" aria-label="Select query patch"></canvas><p id="location"></p><small>Patch indices run left to right, top to bottom. Coordinates refer to this crop.</small></section>
<section><h2>Query → all patches</h2><canvas id="map" width="224" height="224" aria-label="Query relationships"></canvas><p id="scale"></p><p id="cls"></p></section>
<section><h2>All patch pairs</h2><canvas id="matrix" width="256" height="256" aria-label="All patch pairs"></canvas><p>Rows = queries; columns = keys.</p><p id="pair"></p></section>
</div>
<p id="meaning"></p><details><summary>Normalization comparison and sanity checks</summary><pre id="report"></pre></details>
<p>Exact float32 data: <a href="arrays.npz">arrays.npz</a> · Provenance: <a href="report.json">report.json</a>. This dashboard contains the scan crop; review it before sharing.</p>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const payload=JSON.parse(document.getElementById('payload').textContent), arrays={};
for(const [key,b64] of Object.entries(payload.arrays)){const s=atob(b64), b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);const d=new DataView(b.buffer), f=new Float32Array(s.length/4);for(let i=0;i<f.length;i++)f[i]=d.getFloat32(i*4,true);arrays[key]=f;}
const el=id=>document.getElementById(id), photo=new Image();let selectedKey=123;
el('subtitle').textContent=`${payload.report.contract.sample.path || 'Synthetic random noise (seed 42)'} · ${payload.report.contract.backend} · attention block ${payload.report.contract.layer} (0-based) · training input: ${payload.report.training_normalization}`;
el('report').textContent=JSON.stringify({comparison:payload.report.comparison,modes:payload.report.modes,scope:payload.report.scope},null,2);
function color(v,rel,max){if(!Number.isFinite(v))return [140,140,140];if(rel==='attention'){let t=Math.max(0,Math.min(1,v/(max||1)));return [Math.round(245-220*t),Math.round(248-145*t),Math.round(255-55*t)];}let t=Math.min(1,Math.abs(v));return v>=0?[Math.round(245-215*t),Math.round(245-130*t),Math.round(245-40*t)]:[Math.round(245-40*t),Math.round(245-160*t),Math.round(245-195*t)];}
function draw(){const mode=el('normalization').value,rel=el('relation').value,head=el('head').value,q=Math.max(0,Math.min(255,Math.floor(Number(el('query').value)||0)));el('query').value=q;el('head').disabled=rel!=='attention';
 const a=arrays[mode+'_attention'], width=257, heads=head==='mean'?[0,1,2,3,4,5]:[Number(head)];
 function att(i,j){return heads.reduce((s,h)=>s+a[h*width*width+i*width+j],0)/heads.length;}
 function value(i,j){if(rel==='attention')return att(i+1,j+1);const valid=arrays[mode+'_valid_'+rel];return valid[i]&&valid[j]?arrays[mode+'_'+rel][i*256+j]:NaN;}
 let max=0;for(let i=0;i<256;i++)for(let j=0;j<256;j++){const v=value(i,j);if(Number.isFinite(v))max=Math.max(max,v);}const c=el('source').getContext('2d');c.drawImage(photo,0,0,224,224);c.strokeStyle='rgba(255,255,255,.4)';c.lineWidth=.5;for(let i=0;i<=16;i++){c.beginPath();c.moveTo(i*14,0);c.lineTo(i*14,224);c.moveTo(0,i*14);c.lineTo(224,i*14);c.stroke();}c.strokeStyle='#ffbe00';c.lineWidth=2;c.strokeRect((q%16)*14+1,Math.floor(q/16)*14+1,12,12);
 const m=el('map').getContext('2d');for(let j=0;j<256;j++){m.fillStyle=`rgb(${color(value(q,j),rel,max)})`;m.fillRect(j%16*14,Math.floor(j/16)*14,14,14);}m.strokeStyle='#ffbe00';m.lineWidth=2;m.strokeRect(q%16*14+1,Math.floor(q/16)*14+1,12,12);
 const mat=el('matrix').getContext('2d'), img=mat.createImageData(256,256);for(let i=0;i<256;i++)for(let j=0;j<256;j++){const off=(i*256+j)*4;img.data.set([...color(value(i,j),rel,max),255],off);}mat.putImageData(img,0,0);
 el('location').textContent=`Query ${q}: row ${Math.floor(q/16)}, column ${q%16}.`;
 el('scale').textContent=rel==='attention'?`Color scale: 0 → ${max.toPrecision(4)} (maximum of this whole matrix).`:'Fixed scale: orange = −1, white = 0, blue = +1. Gray = undefined for a constant/zero descriptor.';
 el('cls').textContent=rel==='attention'?`Query attention to CLS: ${att(q+1,0).toPrecision(5)}. Patch-only rows are not renormalized; the full 257-token row sums to 1.`:'';
 el('pair').textContent=`Query ${q} → key ${selectedKey}: ${value(q,selectedKey).toPrecision(6)}`;
 el('meaning').textContent=rel==='attention'?'Softmax(QKᵀ / √64) from the selected block, before multiplying by V. The head output A·V is a different tensor. Attention colors rescale per matrix; use exact values for normalization comparisons.':rel==='cosine'?'Cosine compares final LayerNorm patch vectors using their directions. It is symmetric and uses L2 normalization only for this calculation.':'Pearson centers each final patch descriptor across its 384 channels, then computes cosine. It is symmetric; it is not a correlation across patients or proof of anatomical association.';
}
for(const id of ['normalization','relation','head','query'])el(id).addEventListener('change',draw);
function coords(event,canvas,n){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(n-1,Math.floor((event.clientX-r.left)/r.width*n))),Math.max(0,Math.min(n-1,Math.floor((event.clientY-r.top)/r.height*n)))];}
el('source').onclick=e=>{const [x,y]=coords(e,el('source'),16);el('query').value=y*16+x;draw();};
el('map').onclick=e=>{const [x,y]=coords(e,el('map'),16);selectedKey=y*16+x;draw();};
el('matrix').onclick=e=>{const [x,y]=coords(e,el('matrix'),256);el('query').value=y;selectedKey=x;draw();};
photo.onload=draw;photo.src='data:image/png;base64,'+payload.image;
</script></html>'''
