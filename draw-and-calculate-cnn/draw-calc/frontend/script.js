const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const post = (url, body) => fetch(url, {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
let INFO, CHARS = [], SEL = -1, timer = null;

/* ---------- drawing pads (mouse, pen and touch) ---------- */
function makePad(canvas, brush, onEnd){
  const ctx = canvas.getContext("2d");
  let strokes = [], cur = null;
  const pos = e => { const r = canvas.getBoundingClientRect(); return [(e.clientX-r.left)*canvas.width/r.width, (e.clientY-r.top)*canvas.height/r.height]; };
  function redraw(){
    ctx.fillStyle = "#fff"; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle = "#1c1636"; ctx.fillStyle = "#1c1636"; ctx.lineWidth = brush; ctx.lineCap = ctx.lineJoin = "round";
    strokes.forEach(s=>{
      if(s.length === 1){ ctx.beginPath(); ctx.arc(s[0][0], s[0][1], brush/2, 0, 7); ctx.fill(); return; }
      ctx.beginPath(); ctx.moveTo(...s[0]); s.slice(1).forEach(p=>ctx.lineTo(...p)); ctx.stroke();
    });
  }
  canvas.addEventListener("pointerdown", e=>{ canvas.setPointerCapture(e.pointerId); cur = [pos(e)]; strokes.push(cur); redraw(); onEnd && onEnd("start"); });
  canvas.addEventListener("pointermove", e=>{ if(!cur) return; cur.push(pos(e)); redraw(); });
  const end = ()=>{ if(cur){ cur = null; onEnd && onEnd("end"); } };
  canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);
  redraw();
  return {undo(){ strokes.pop(); redraw(); }, clear(){ strokes = []; redraw(); }, empty:()=>strokes.length===0, image:()=>canvas.toDataURL("image/png")};
}

/* ---------- 1. calculator ---------- */
const pad = makePad($("pad"), 9, ev=>{
  $("padhint").style.display = "none";
  if(ev === "start") clearTimeout(timer);
  if(ev === "end" && $("auto").checked){ clearTimeout(timer); timer = setTimeout(read, 900); }
});
$("btnCalc").onclick = read;
$("btnUndo").onclick = ()=>{ pad.undo(); if(pad.empty()) resetOut(); else if($("auto").checked) read(); };
$("btnClear").onclick = ()=>{ pad.clear(); resetOut(); $("padhint").style.display = ""; };
function resetOut(){ $("out").hidden = true; CHARS = []; SEL = -1; $("inspect").hidden = true; $("inspectEmpty").hidden = false; }

async function read(){
  if(pad.empty()){ return; }
  const r = await post("/api/read", {image: pad.image()}); const d = await r.json();
  $("out").hidden = false;
  if(!r.ok){ $("lcd").innerHTML = "&nbsp;"; $("msg").textContent = d.error; $("chars").innerHTML = ""; $("stepsBox").hidden = true; return; }
  CHARS = d.chars.map(c=>({...c}));
  drawChars(); showCalc(d.calc, d.calc_error);
  const first = Math.max(0, CHARS.findIndex(c=>c.uncertain)); inspect(first);
}

function drawChars(){
  const all = INFO.classes;
  $("chars").innerHTML = CHARS.map((c,i)=>{
    const alts = c.top.map(t=>t.char), rest = all.filter(x=>!alts.includes(x));
    const opts = [...alts, ...rest].map(x=>`<option ${x===c.char?"selected":""}>${x}</option>`).join("");
    return `<div class="ch ${c.uncertain?"unsure":""} ${i===SEL?"sel":""}" data-i="${i}"><canvas width="28" height="28"></canvas>
      <div class="big">${c.char}</div><div class="conf">${c.corrected?"fixed by you":c.uncertain?"not sure "+Math.round(c.confidence*100)+"%":Math.round(c.confidence*100)+"%"}</div>
      <select data-i="${i}">${opts}</select></div>`;
  }).join("");
  document.querySelectorAll("#chars .ch").forEach(el=>{
    paint(el.querySelector("canvas"), CHARS[el.dataset.i].pixels, 28);
    el.onclick = e=>{ if(e.target.tagName !== "SELECT") inspect(+el.dataset.i); };
    el.querySelector("select").onchange = async e=>{
      const i = +e.target.dataset.i; CHARS[i].char = e.target.value; CHARS[i].corrected = true; CHARS[i].uncertain = false;
      const r = await post("/api/calculate", {chars: CHARS.map(c=>c.char)}); const d = await r.json();
      drawChars(); r.ok ? showCalc(d) : showCalc(null, d.error);
    };
  });
}

function showCalc(calc, err){
  const expr = CHARS.map(c=>"+-×÷=".includes(c.char) ? " "+c.char+" " : c.char).join("").trim().replace(/\s*=\s*$/, "");
  if(calc){
    $("lcd").innerHTML = `${esc(expr)} = <span class="ans">${esc(calc.result)}</span>`; $("msg").textContent = "";
    $("stepsBox").hidden = calc.steps.length === 0;
    $("steps").innerHTML = calc.steps.map(s=>`<li>${esc(s.text)} <span>→ ${esc(s.left)}</span></li>`).join("");
  } else {
    $("lcd").textContent = expr; $("msg").textContent = err; $("stepsBox").hidden = true;
  }
}

function paint(canvas, px, n, scaleMax){
  const ctx = canvas.getContext("2d"), img = ctx.createImageData(n, n);
  for(let i=0;i<n*n;i++){ const v = Array.isArray(px[0]) ? px.flat()[i] : px[i]; img.data[i*4]=img.data[i*4+1]=img.data[i*4+2]=v; img.data[i*4+3]=255; }
  ctx.putImageData(img, 0, 0);
}

/* ---------- 2. inside the CNN ---------- */
async function inspect(i){
  if(!CHARS[i]) return; SEL = i;
  document.querySelectorAll("#chars .ch").forEach(el=>el.classList.toggle("sel", +el.dataset.i === i));
  const d = await (await post("/api/inspect", {pixels: CHARS[i].pixels})).json();
  $("inspect").hidden = false; $("inspectEmpty").hidden = true;
  paint($("inImg"), CHARS[i].pixels, 28);
  const mk = (id, maps, n)=>{ $(id).innerHTML = maps.map(()=>`<canvas width="${n}" height="${n}"></canvas>`).join(""); $(id).querySelectorAll("canvas").forEach((c,k)=>paint(c, maps[k], n)); };
  mk("m1", d.conv1, 26); mk("m2", d.conv2, 11);
  const top = Object.entries(d.probs).sort((a,b)=>b[1]-a[1]);
  $("probs").innerHTML = top.slice(0,6).map(([c,p],k)=>`<div class="prow ${k===0?"top":""}"><span>${c}</span><i><b style="width:${p*100}%"></b></i><span>${(p*100).toFixed(1)}%</span></div>`).join("");
}

/* ---------- 3. training results ---------- */
function drawTraining(){
  const m = INFO, t = m.test, c = m.counts;
  $("arch").innerHTML = [["Input","1x28x28"],["Conv2D","8 filters 3x3, ReLU"],["MaxPool","2x2"],["Conv2D","16 filters 3x3, ReLU"],["MaxPool","2x2"],["Flatten","400"],["Dense","64, ReLU"],["Dense","15, Softmax"]]
    .map(l=>`<div class="layer"><b>${l[0]}</b>${l[1]}</div>`).join('<span class="arrow">→</span>');
  const pct = v=>(v*100).toFixed(1)+"%";
  $("stats").innerHTML = [[pct(t.real_digits_acc),`accuracy on ${t.real_digits_n} real handwritten digits never seen in training`],[pct(t.overall_acc),"accuracy on the whole test set"],
    [pct(t.shifted_noisy_acc),"accuracy on shifted + noisy test images"],[c.train.toLocaleString(),"training images"],[m.params.toLocaleString(),"learnable parameters"],[m.best_epoch+" / "+m.epochs,"best epoch (lowest validation loss)"]]
    .map(s=>`<div class="stat"><b>${s[0]}</b><span>${s[1]}</span></div>`).join("");
  $("honest").textContent = `Data: ${c.generated.toLocaleString()} generated characters (digits and operators) plus ${c.real_digits.toLocaleString()} real handwritten digits. The real-digit score is the most honest one, because the generated test images come from the same generator as the training images, so they are easier.`;

  const ep = m.history.loss.map((_,i)=>i+1), grid = {color:"#ece8fb"};
  const line = (id,a,b,yf,min)=> new Chart($(id),{type:"line",data:{labels:ep,datasets:[
      {label:"Training",data:a,borderColor:"#1c1636",borderWidth:2.5,pointRadius:0,tension:.25},{label:"Validation",data:b,borderColor:"#ff5d73",borderWidth:2.5,pointRadius:0,tension:.25}]},
    options:{responsive:true,maintainAspectRatio:false,interaction:{mode:"index",intersect:false},scales:{x:{title:{display:true,text:"Epoch"},grid:{display:false}},y:{min,grid,ticks:{callback:yf}}},plugins:{legend:{position:"bottom"}}}});
  line("cLoss", m.history.loss, m.history.val_loss, v=>v, 0);
  line("cAcc", m.history.acc.map(v=>v*100), m.history.val_acc.map(v=>v*100), v=>v+"%", 90);

  const K = m.classes, cm = m.confusion;
  $("cm").innerHTML = `<tr><th></th>${K.map(k=>`<th>${k}</th>`).join("")}</tr>` + cm.map((row,i)=>{
    const tot = row.reduce((a,b)=>a+b,0)||1;
    return `<tr><th>${K[i]}</th>${row.map((v,j)=>{ const a = v/tot; const bg = v===0 ? "#f6f4ff" : i===j ? `rgba(61,220,151,${.25+.75*a})` : `rgba(255,93,115,${.4+.6*a})`; return `<td style="background:${bg}">${v||""}</td>`; }).join("")}</tr>`;
  }).join("");
  $("perclass").innerHTML = K.map(k=>`<div class="pc"><span>${k}</span><i><b style="width:${m.per_class[k]*100}%"></b></i><span>${(m.per_class[k]*100).toFixed(1)}%</span></div>`).join("");
  $("pairs").innerHTML = m.confused_pairs.length ? m.confused_pairs.map(p=>`<li>Real <b>${p.true}</b> read as <b>${p.pred}</b>: ${p.count} time${p.count>1?"s":""}</li>`).join("") : "<li>No mistakes on the test set.</li>";
  $("gallery").innerHTML = K.map(k=>`<div class="grow"><b>${k}</b>${m.gallery[k].map(()=>`<canvas width="28" height="28"></canvas>`).join("")}</div>`).join("");
  document.querySelectorAll("#gallery .grow").forEach((row,i)=>row.querySelectorAll("canvas").forEach((cv,j)=>paint(cv, m.gallery[K[i]][j], 28)));
}

/* ---------- 4. my handwriting ---------- */
let target = "";
const pad2 = makePad($("pad2"), 16);
function newTarget(){ let t; do { t = INFO.classes[Math.floor(Math.random()*INFO.classes.length)]; } while(t === target); target = t; $("target").textContent = t; pad2.clear(); $("verdict").textContent = ""; $("verdict").className = "verdict"; }
$("btnNew").onclick = newTarget;
$("btnClear2").onclick = ()=>pad2.clear();
$("btnCheck").onclick = async ()=>{
  const r = await post("/api/selftest", {image: pad2.image(), label: target}); const d = await r.json();
  const v = $("verdict");
  if(!r.ok){ v.textContent = d.error; v.className = "verdict bad"; return; }
  v.textContent = d.right ? `Correct: the CNN read ${d.pred} (${Math.round(d.conf*100)}%)` : `The CNN read ${d.pred} (${Math.round(d.conf*100)}%), you drew ${target}`;
  v.className = "verdict " + (d.right ? "ok" : "bad"); drawTally(d.tally);
  setTimeout(newTarget, 1400);
};
function drawTally(t){
  $("tally").innerHTML = `<div class="stat"><b>${t.total}</b><span>drawings tested</span></div><div class="stat"><b>${t.correct}</b><span>read correctly</span></div><div class="stat"><b>${t.accuracy===null?"-":(t.accuracy*100).toFixed(0)+"%"}</b><span>my accuracy</span></div>`;
  $("tallyChars").innerHTML = Object.entries(t.per_char).map(([k,v])=>`<span>${k} ${v.correct}/${v.n}</span>`).join("") || "<span>none yet</span>";
  $("recent").innerHTML = t.recent.map(r=>`<li>drew <b>${r.label}</b>, CNN said <b>${r.pred}</b> ${r.label===r.pred?"(right)":"(wrong)"}</li>`).join("") || "<li>No answers yet.</li>";
}

(async function init(){
  INFO = await (await fetch("/api/info")).json();
  drawTraining(); newTarget(); drawTally(await (await fetch("/api/selftest")).json());
})();
