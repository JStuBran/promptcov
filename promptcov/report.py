"""Single-file HTML report. The prompt itself is the hero: rendered as a
document on a lightbox, every segment stamped with its verdict. The
coverage strip on top is the shareable minimap."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import stats as st


def _payload(res) -> dict:
    segs = []
    for b in res.doc.blocks:
        v = res.verdicts.get(b.id)
        item = {"id": b.id, "kind": b.kind, "section": b.section_id,
                "text": b.text, "verdict": (v.verdict if v else None),
                "label": (st.VERDICT_LABEL.get(v.verdict) if v else None),
                "note": (v.note if v else ""),
                "pruned": b.id in set(res.pruned_ids)}
        if v:
            for name in ("deletion", "negation", "probe"):
                tr = getattr(v, name)
                item[name] = tr.summary() if tr else None
            item["example"] = v.example
        segs.append(item)

    leaf_chars = sum(len(b.text) for b in res.doc.blocks if b.kind == "leaf")
    dead_chars = sum(len(b.text) for b in res.doc.blocks
                     if b.kind == "leaf" and res.verdicts.get(b.id) and
                     res.verdicts[b.id].verdict in
                     (st.NO_OBSERVED_EFFECT, st.INHERITED))
    counts: dict[str, int] = {}
    for v in res.verdicts.values():
        counts[v.verdict] = counts.get(v.verdict, 0) + 1

    import statistics
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "meta": res.meta,
        "noise": {"median": round(statistics.median(res.noise), 4)
                  if res.noise else 0,
                  "p95": round(st.percentile(res.noise, 0.95), 4)},
        "segments": segs,
        "counts": counts,
        "dead_pct": round(100 * dead_chars / max(1, leaf_chars), 1),
        "verification": res.verification,
        "kept_redundant": res.kept_redundant,
    }


def payload(res) -> dict:
    """Machine-readable verdicts — same data the HTML report embeds.
    Diff two of these across runs, models, or prompt versions."""
    return _payload(res)


def render(res, title: str = "system prompt") -> str:
    data = json.dumps(_payload(res)).replace("</", "<\\/")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>promptcov · __TITLE__</title>
<!-- no external requests: the report stays a single self-contained file.
     Space Grotesk / IBM Plex Mono are used when installed locally, with
     system fallbacks otherwise. -->
<style>
:root{
  --chrome:#14181d; --chrome-2:#1b2129; --line:#2a323d;
  --ink-on-dark:#c9d3de; --dim:#7d8b99;
  --paper:#f2ede3; --paper-edge:#e3dccb; --ink:#20252d;
  --load:#c8502e; --load-soft:#c8502e2b;
  --ghost:#9a938433;
  --redun:#7b5cd6; --redun-soft:#7b5cd626;
  --dorm:#1f8a8a; --dorm-soft:#1f8a8a22;
  --ok:#3e8e5a; --fail:#c8502e;
}
*{box-sizing:border-box;margin:0}
body{background:var(--chrome);color:var(--ink-on-dark);
  font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:13px;line-height:1.55}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 80px}
header h1{font-family:"Space Grotesk",system-ui,sans-serif;font-weight:700;
  font-size:clamp(26px,4vw,40px);letter-spacing:-.5px;color:#fff}
header h1 em{font-style:normal;color:var(--load)}
.sub{color:var(--dim);margin-top:2px}
.bignum{font-family:"Space Grotesk",system-ui,sans-serif;font-weight:700;
  font-size:clamp(40px,7vw,72px);color:#fff;line-height:1;margin-top:22px}
.bignum small{font-size:16px;font-weight:400;color:var(--dim);display:block;
  margin-top:6px;max-width:560px;font-family:"IBM Plex Mono",monospace}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
.chip{border:1px solid var(--line);border-radius:4px;padding:3px 9px;
  color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.08em}
/* coverage strip — the signature */
.striplabel{margin:26px 0 6px;color:var(--dim);font-size:11px;
  text-transform:uppercase;letter-spacing:.12em}
.strip{display:flex;gap:2px;height:26px;border:1px solid var(--line);
  border-radius:4px;padding:3px;background:var(--chrome-2)}
.strip .cell{flex:1;border-radius:2px;min-width:3px;cursor:pointer;
  opacity:0;animation:pop .35s forwards}
@keyframes pop{to{opacity:1}}
@media (prefers-reduced-motion: reduce){.strip .cell{animation:none;opacity:1}}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:10px 0 26px;color:var(--dim);font-size:11.5px}
.legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
/* two-pane */
.panes{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(280px,1fr);gap:20px}
@media (max-width:900px){.panes{grid-template-columns:1fr}}
.paper{background:var(--paper);color:var(--ink);border-radius:6px;
  border:1px solid var(--paper-edge);
  box-shadow:0 0 0 1px #000 inset,0 18px 50px -20px #000c;
  padding:26px 26px 34px;white-space:pre-wrap;word-wrap:break-word;
  font-size:12.5px;line-height:1.7}
.paper .hdr{font-weight:700}
.seg{border-radius:2px;padding:0 1px;cursor:pointer;transition:filter .12s}
.seg:hover{filter:brightness(.93)}
.v-LOAD_BEARING{background:var(--load-soft);box-shadow:inset 3px 0 0 var(--load)}
.v-NO_OBSERVED_EFFECT,.v-NO_OBSERVED_EFFECT_VIA_SECTION{
  color:#8b8577;background:var(--ghost);
  text-decoration:line-through;text-decoration-color:#8b857766;text-decoration-thickness:1px}
.v-REDUNDANT{background:var(--redun-soft);box-shadow:inset 3px 0 0 var(--redun)}
.v-UNEXERCISED{background:var(--dorm-soft);
  border-bottom:2px dashed var(--dorm)}
.seg.sel{outline:2px solid #20252d;outline-offset:1px}
/* inspector */
.inspector{position:sticky;top:16px;align-self:start;background:var(--chrome-2);
  border:1px solid var(--line);border-radius:6px;padding:18px;min-height:220px}
.inspector h3{font-family:"Space Grotesk",sans-serif;color:#fff;font-size:15px;margin-bottom:2px}
.badge{display:inline-block;border-radius:3px;padding:2px 8px;font-size:11px;
  letter-spacing:.06em;text-transform:uppercase;margin:8px 0 12px;color:#fff}
.b-LOAD_BEARING{background:var(--load)} .b-REDUNDANT{background:var(--redun)}
.b-UNEXERCISED{background:var(--dorm)}
.b-NO_OBSERVED_EFFECT,.b-NO_OBSERVED_EFFECT_VIA_SECTION{background:#4a5563}
.stat{display:flex;justify-content:space-between;border-top:1px dashed var(--line);
  padding:6px 0;color:var(--ink-on-dark)}
.stat span:first-child{color:var(--dim)}
.note{color:var(--dim);font-size:12px;margin-top:10px;font-style:italic}
.ex{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}
.ex h4{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.1em;margin:8px 0 4px}
.ex pre{white-space:pre-wrap;background:var(--chrome);border:1px solid var(--line);
  border-radius:4px;padding:8px;font-size:11.5px;max-height:130px;overflow:auto}
.ex .b pre{border-left:3px solid #4a5563}.ex .v pre{border-left:3px solid var(--load)}
/* verification */
.verify{margin-top:26px;border:1px solid var(--line);border-radius:6px;
  background:var(--chrome-2);padding:18px}
.verify h2, .foot h2{font-family:"Space Grotesk",sans-serif;color:#fff;font-size:16px;margin-bottom:8px}
.pass{color:var(--ok);font-weight:700}.failtxt{color:var(--fail);font-weight:700}
.foot{margin-top:26px;color:var(--dim);border-top:1px solid var(--line);padding-top:16px}
.foot p{max-width:760px;margin-bottom:8px}
.foot .tattoo{color:#e8d9b0}
a{color:var(--dorm)}
</style></head><body><div class="wrap">
<header>
  <h1>prompt<em>cov</em></h1>
  <div class="sub">coverage report · __TITLE__ · <span id="gen"></span></div>
  <div class="bignum"><span id="deadpct"></span>%
    <small>of this prompt's rule text showed <b>no observed effect on this
    traffic distribution</b>. Not "useless" — unobserved. Read the footer
    before you delete anything.</small></div>
  <div class="chips" id="chips"></div>
</header>

<div class="striplabel">coverage strip — one cell per segment, in document order</div>
<div class="strip" id="strip"></div>
<div class="legend">
  <span><b style="background:var(--load)"></b>load-bearing</span>
  <span><b style="background:#8b8577"></b>no observed effect</span>
  <span><b style="background:var(--redun)"></b>redundant</span>
  <span><b style="background:var(--dorm)"></b>unexercised, exercisable</span>
  <span><b style="background:#3a424d"></b>header / untested</span>
</div>

<div class="panes">
  <div class="paper" id="paper"></div>
  <aside class="inspector" id="inspector">
    <h3>Inspector</h3>
    <div class="note">Click any highlighted segment — in the document or the
    strip — to see its deletion, negation, and probe statistics with a
    sample divergence.</div>
  </aside>
</div>

<div class="verify" id="verify"></div>

<div class="foot">
  <h2>How to read this</h2>
  <p>Every verdict is measured against the prompt's own <b>noise floor</b>
  (median <span id="nmed"></span>, p95 <span id="np95"></span>): the unchanged
  prompt was run <span id="reps"></span>× and its self-drift is the null
  hypothesis. A segment counts as load-bearing only when ablating it moves
  outputs beyond that drift with p &lt; <span id="alpha"></span> (permutation
  test) and a minimum effect size.</p>
  <p class="tattoo">⚠ Unexercised ≠ useless. A rule can show zero coverage
  because your traffic is polite — the same way an error handler shows 0%
  test coverage until the day it saves you. This report says "no observed
  effect on this distribution," never "dead." Safety rules especially:
  probe them before you touch them.</p>
  <p>Redundant = deleting it changes nothing but inverting it does; it's
  covered by a sibling. Prune groups by hand, one survivor per group.</p>
</div>
</div>
<script>
const D = __DATA__;
document.getElementById('gen').textContent = D.generated;
document.getElementById('deadpct').textContent = D.dead_pct;
document.getElementById('nmed').textContent = D.noise.median;
document.getElementById('np95').textContent = D.noise.p95;
document.getElementById('reps').textContent = D.meta.replicates;
document.getElementById('alpha').textContent = D.meta.alpha;
const chips = [
  D.meta.provider + " · " + D.meta.model,
  D.meta.inputs + " traffic inputs",
  D.meta.replicates + " baseline replicates",
  D.meta.provider_calls + " model calls",
  (D.meta.negate?"negation ✓":"negation –"),
  (D.meta.probes?"probes ✓":"probes –"),
  D.meta.seconds + "s"
];
document.getElementById('chips').innerHTML =
  chips.map(c=>`<div class="chip">${c}</div>`).join('');

const paper = document.getElementById('paper');
const strip = document.getElementById('strip');
const insp  = document.getElementById('inspector');
const esc = s => s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const CELL = {LOAD_BEARING:'var(--load)', NO_OBSERVED_EFFECT:'#8b8577',
  NO_OBSERVED_EFFECT_VIA_SECTION:'#8b8577', REDUNDANT:'var(--redun)',
  UNEXERCISED:'var(--dorm)'};

D.segments.forEach((s,i)=>{
  const el = document.createElement('span');
  el.id = 'seg-'+i;
  if(s.kind==='header'){ el.className='hdr'; el.textContent=s.text; }
  else{
    el.className='seg'+(s.verdict?' v-'+s.verdict:'');
    el.textContent=s.text;
    el.onclick=()=>select(i);
  }
  paper.appendChild(el);
  const c=document.createElement('div');
  c.className='cell';
  c.style.background = s.kind==='header' ? '#3a424d'
    : (CELL[s.verdict]||'#3a424d');
  c.style.animationDelay=(i*18)+'ms';
  c.title=(s.verdict? s.label+': ':'')+s.text.trim().slice(0,80);
  c.onclick=()=>{select(i);document.getElementById('seg-'+i)
    .scrollIntoView({block:'center',behavior:'smooth'});};
  strip.appendChild(c);
});

function row(k,v){return `<div class="stat"><span>${k}</span><span>${v}</span></div>`}
function trRows(name,t){
  if(!t) return '';
  return row(name+' effect', (t.effect>0?'+':'')+t.effect) +
         row(name+' p-value', t.p) +
         row(name+' inputs > noise p95', Math.round(t.exceed_p95*100)+'%');
}
function select(i){
  document.querySelectorAll('.seg.sel').forEach(e=>e.classList.remove('sel'));
  const el=document.getElementById('seg-'+i); if(el) el.classList.add('sel');
  const s=D.segments[i];
  let h=`<h3>${s.id}</h3>`;
  h+=`<span class="badge b-${s.verdict}">${s.label||'untested'}</span>`;
  h+=trRows('deletion',s.deletion)+trRows('negation',s.negation)+trRows('probe',s.probe);
  if(s.pruned) h+=row('pruned prompt','removed ✂');
  if(s.note) h+=`<div class="note">${esc(s.note)}</div>`;
  if(s.example && s.example.input){
    h+=`<div class="ex"><h4>most divergent input (Δ ${s.example.divergence})</h4>
    <pre>${esc(s.example.input)}</pre>
    <div class="b"><h4>baseline</h4><pre>${esc(s.example.baseline)}</pre></div>
    <div class="v"><h4>ablated</h4><pre>${esc(s.example.variant)}</pre></div></div>`;
  }
  insp.innerHTML=h;
}

const V=D.verification;
document.getElementById('verify').innerHTML =
 `<h2>Verified pruned prompt</h2>
  <div class="stat"><span>size</span><span>${V.original_chars} → ${V.pruned_chars} chars (−${V.reduction_pct}%)</span></div>
  <div class="stat"><span>full-corpus regression test</span>
    <span class="${V.passed?'pass':'failtxt'}">${V.passed?'PASS — evals green':'FAIL'}</span></div>
  ${V.rescued.length?`<div class="stat"><span>rescued during verification</span><span>${V.rescued.map(r=>r.segment).join(', ')}</span></div>`:''}
  ${D.kept_redundant.length?`<div class="stat"><span>kept (redundant — consolidate by hand)</span><span>${D.kept_redundant.join(', ')}</span></div>`:''}
  ${V.regressed_inputs.length?`<div class="note">Regressing inputs: ${V.regressed_inputs.map(esc).join(' · ')}</div>`:''}`;
</script></body></html>
"""
