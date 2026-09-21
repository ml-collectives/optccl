"""Turn an OptCCL schedule into a self-contained interactive HTML page.

Usage (runnable as-is from the repo root):

    optccl run all_gather tools/examples/ring_plain.json 1 \
        --write-schedule -o plain.json
    python tools/visualize_schedule.py plain.json \
        --topology tools/examples/ring_plain.json -o viz.html

Several schedules render side by side against the same topology, which is the
point when comparing (say) a pass-through switch against a multicast one:

    optccl run all_gather tools/examples/ring_mcast.json 1 \
        --write-schedule -o mcast.json
    python tools/visualize_schedule.py plain.json mcast.json \
        --topology tools/examples/ring_plain.json \
        --label "Pass-through" --label "Multicast" -o cmp.html

(Both ring specs share a layout, so either works as --topology; the multicast
one differs only in the switch's capabilities.)

The pipeline is deliberately split so other renderers can reuse the front half:

    load_schedule()  -> per-step link activity, straight off the schedule JSON
    layout()         -> node positions via networkx
    build_payload()  -> the renderer-agnostic intermediate form
    render_html()    -> one possible renderer (canvas flow animation)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optccl.config import Config  # noqa: E402
from optccl.topology_spec import load_or_build_topology  # noqa: E402


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def node_id(n) -> str:
    """Render a component id as a short stable string: ('g', 1) -> 'g1'."""
    if isinstance(n, (list, tuple)):
        return "".join(str(p) for p in n)
    return str(n)


def load_schedule(path: str) -> dict:
    """Read a `--write-schedule` result and collapse chunks into per-link volume.

    The schedule stores every chunk individually; for a picture we only need,
    per step, how much each link moved and where it was headed.
    """
    with open(path) as fh:
        data = json.load(fh)

    if "schedule" not in data:
        raise SystemExit(
            f"{path}: no 'schedule' key -- re-run with --write-schedule"
        )

    steps = []
    for key in sorted(data["schedule"], key=int):
        entry = data["schedule"][key]
        agg = defaultdict(lambda: {"kB": 0.0, "dests": set()})
        for edge_key, txs in entry["transmissions"].items():
            src, dst = ast.literal_eval(edge_key)
            for _chunk_no, size, switches, dests in txs:
                via = tuple(node_id(s) for s in switches)
                rec = agg[(node_id(src), node_id(dst), via)]
                rec["kB"] += size
                rec["dests"].update(node_id(d) for d in dests)
        links = [
            {"src": s, "dst": d, "via": list(v), "kB": r["kB"],
             "dests": sorted(r["dests"])}
            for (s, d, v), r in agg.items()
        ]
        links.sort(key=lambda L: -L["kB"])
        steps.append({"t": float(entry.get("start_time", 0.0)), "links": links})

    return {
        "steps": steps,
        "objective": data.get("objective_value"),
        "latency": data.get("measured_latency_s"),
        "bw": data.get("algorithmic_bandwidth_GBps"),
        "overhead": data.get("overhead"),
        "n_steps": data.get("n_steps", len(steps)),
    }


def step_durations(steps: list, total) -> list:
    """Per-step wall time, from consecutive start times."""
    out = []
    for i, s in enumerate(steps):
        if i + 1 < len(steps):
            out.append(max(steps[i + 1]["t"] - s["t"], 1e-9))
        else:
            tail = (total - s["t"]) if total else 0.0
            out.append(max(tail, out[-1] if out else 1e-9))
    return out


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def layout(top, kind: str = "shell", size: int = 640, pad: int = 56) -> dict:
    """Position every component. GPUs ring the outside, everything else inside."""
    g = nx.Graph()
    g.add_nodes_from(node_id(c) for c in top.components)
    for (u, v) in top.edge_data:
        if u != v:
            g.add_edge(node_id(u), node_id(v))

    gpus = [node_id(c) for c in top.gpus]
    inner = [n for n in g.nodes if n not in set(gpus)]

    if kind == "spring":
        pos = nx.spring_layout(g, seed=7)
    elif kind == "circular":
        pos = nx.circular_layout(g)
    elif inner:
        pos = nx.shell_layout(g, nlist=[inner, gpus])
    else:
        pos = nx.circular_layout(g)

    xs = [p[0] for p in pos.values()] or [0.0]
    ys = [p[1] for p in pos.values()] or [0.0]
    lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
    span = max(hi_x - lo_x, hi_y - lo_y, 1e-9)
    usable = size - 2 * pad

    out = {}
    for n, (x, y) in pos.items():
        out[n] = [
            round(pad + (x - lo_x) / span * usable, 2),
            round(pad + (hi_y - y) / span * usable, 2),  # flip: SVG y grows down
        ]
    return out


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------


def build_payload(runs: list, top, pos: dict, size: int) -> dict:
    """The renderer-agnostic intermediate form."""
    gpus = {node_id(c) for c in top.gpus}
    kinds, fan = {}, {}
    for c in top.components:
        caps = set(top.capabilities[c])
        nid = node_id(c)
        if nid in gpus:
            kinds[nid] = "gpu"
        elif "storage" in caps:
            kinds[nid] = "mem"
        else:
            kinds[nid] = "switch"
        # Can this non-GPU duplicate or combine data? Worth marking: it is the
        # difference between a wire and a multicast switch.
        fan[nid] = nid not in gpus and bool({"copy", "reduce"} & caps)

    base_edges = sorted(
        {
            tuple(sorted((node_id(u), node_id(v))))
            for (u, v) in top.edge_data
            if u != v
        }
    )

    out_runs = []
    for r in runs:
        peak = max(
            (L["kB"] for s in r["steps"] for L in s["links"]), default=1.0
        ) or 1.0
        out_runs.append({
            "label": r["label"],
            "objective": r["objective"],
            "latency": r["latency"],
            "bw": r["bw"],
            "overhead": r["overhead"],
            "peak": peak,
            "dur": [round(d, 6) for d in step_durations(r["steps"], r["latency"])],
            "steps": [
                {
                    "t": round(s["t"], 6),
                    "links": [
                        {
                            "s": L["src"], "d": L["dst"], "v": L["via"],
                            "kB": round(L["kB"], 1), "to": L["dests"],
                        }
                        for L in s["links"]
                    ],
                }
                for s in r["steps"]
            ],
        })

    return {
        "size": size,
        "nodes": [
            {"id": n, "kind": kinds.get(n, "switch"), "fan": bool(fan.get(n)),
             "x": pos[n][0], "y": pos[n][1]}
            for n in sorted(pos)
        ],
        "edges": [list(e) for e in base_edges],
        "runs": out_runs,
    }


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------

_TEMPLATE = r"""<title>__TITLE__</title>
<style>
  :root {
    --bg:#ECEFF3; --surface:#FFF; --surface-2:#F2F5F8; --ink:#101922; --muted:#59687A;
    --faint:#90A0AC; --line:#CCD5DD; --ring:#A94F0C; --switch:#0A6C83; --mem:#6B3FA0;
    --gpu-fill:#1D2933; --gpu-ink:#FFF; --shadow:0 1px 2px rgba(16,25,34,.06),0 8px 24px rgba(16,25,34,.05);
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  }
  @media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
    --bg:#0E1319; --surface:#151C24; --surface-2:#1A232C; --ink:#E2EAF1; --muted:#8393A0;
    --faint:#596874; --line:#232D37; --ring:#E8912F; --switch:#2BC2DE; --mem:#B48BE8;
    --gpu-fill:#E2EAF1; --gpu-ink:#0E1319; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
  }}
  :root[data-theme="dark"]{
    --bg:#0E1319; --surface:#151C24; --surface-2:#1A232C; --ink:#E2EAF1; --muted:#8393A0;
    --faint:#596874; --line:#232D37; --ring:#E8912F; --switch:#2BC2DE; --mem:#B48BE8;
    --gpu-fill:#E2EAF1; --gpu-ink:#0E1319; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.3);
  }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5;margin:0;
       padding:clamp(18px,3.5vw,40px) clamp(14px,3.5vw,36px) 56px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:22px}
  .eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.16em;text-transform:uppercase;
           color:var(--muted);margin:0 0 8px}
  h1{font-size:clamp(25px,3.6vw,34px);letter-spacing:-.024em;line-height:1.12;margin:0 0 10px;
     font-weight:620;text-wrap:balance}
  .lede{color:var(--muted);max-width:68ch;margin:0;font-size:15px}
  .panels{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
  .panel{background:var(--surface);border:1px solid var(--line);border-radius:10px;
         box-shadow:var(--shadow);padding:16px;display:flex;flex-direction:column;gap:12px}
  .phead{display:flex;align-items:baseline;justify-content:space-between;gap:10px;flex-wrap:wrap}
  .ptitle{font-size:15px;font-weight:640;margin:0;letter-spacing:-.01em}
  .stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);
         border:1px solid var(--line);border-radius:7px;overflow:hidden}
  .stat{background:var(--surface-2);padding:7px 9px}
  .stat dt{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
           color:var(--muted);margin:0}
  .stat dd{font-family:var(--mono);font-size:14.5px;font-variant-numeric:tabular-nums;margin:0;
           letter-spacing:-.02em}
  .stat dd .u{font-size:10px;color:var(--muted);margin-left:2px}
  canvas{width:100%;height:auto;display:block;border-radius:8px}
  .now{font-family:var(--mono);font-size:11.5px;color:var(--muted);border-top:1px dashed var(--line);
       padding-top:9px;min-height:54px}
  .now b{color:var(--ink)}
  .now b.done{color:var(--faint);font-weight:500}
  .now .row{display:flex;gap:7px;flex-wrap:wrap;margin-top:5px}
  .tag{border:1px solid var(--line);border-radius:4px;padding:1px 6px;background:var(--surface-2);
       white-space:nowrap}
  .tag.sw{color:var(--switch)} .tag.rg{color:var(--ring)}
  .transport{background:var(--surface);border:1px solid var(--line);border-radius:10px;
             box-shadow:var(--shadow);padding:13px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  button{font-family:var(--mono);font-size:12px;color:var(--ink);background:var(--surface-2);
         border:1px solid var(--line);border-radius:6px;padding:7px 13px;cursor:pointer}
  button:hover{border-color:var(--faint)}
  button:focus-visible,input:focus-visible{outline:2px solid var(--switch);outline-offset:2px}
  button.primary{background:var(--ink);color:var(--bg);border-color:var(--ink);min-width:76px}
  input[type=range]{flex:1;min-width:150px;accent-color:var(--switch)}
  .ro{font-family:var(--mono);font-size:12.5px;font-variant-numeric:tabular-nums;color:var(--muted);
      white-space:nowrap}
  .ro b{color:var(--ink)}
  select{font-family:var(--mono);font-size:12px;background:var(--surface-2);color:var(--ink);
         border:1px solid var(--line);border-radius:6px;padding:6px 8px}
  .legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12.5px;color:var(--muted);align-items:center}
  .sw{width:16px;height:3px;border-radius:2px;display:inline-block;vertical-align:middle;margin-right:6px}
  footer{color:var(--faint);font-size:12px;font-family:var(--mono);border-top:1px solid var(--line);
         padding-top:12px}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">__EYEBROW__</p>
    <h1>__HEADING__</h1>
    <p class="lede">Each moving dot is data in flight. Dots travel at a fixed speed &mdash; that's the
      link rate &mdash; so a busier link carries <em>more</em> dots, not faster ones. Line thickness is
      the same signal. Playback runs in schedule time, so longer steps genuinely take longer.</p>
  </header>
  <div class="panels" id="panels"></div>
  <div class="transport">
    <button class="primary" id="play">Play</button>
    <button id="prev">&larr;</button>
    <button id="next">&rarr;</button>
    <input type="range" id="scrub" min="0" step="0.001" value="0" aria-label="Schedule time" />
    <span class="ro">moment <b id="sn">0</b><span id="st"></span> &middot;
      t=<b id="tt">0.000</b>s &middot; <b id="live">-</b> still running</span>
    <label class="ro">speed
      <select id="speed">
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option>
      </select>
    </label>
  </div>
  <div class="legend">
    <span><i class="sw" style="background:var(--ring)"></i>direct link</span>
    <span><i class="sw" style="background:var(--switch)"></i>through a switch</span>
    <span>thickness &amp; dot count = volume</span>
    <span style="font-family:var(--mono);font-size:11.5px">&#9647; outlined = can copy/reduce</span>
  </div>
  <footer id="foot"></footer>
</div>

<script type="application/json" id="payload">__DATA__</script>
<script>
(function(){
"use strict";
var P = JSON.parse(document.getElementById("payload").textContent);
var NODES = {}, SIZE = P.size;
P.nodes.forEach(function(n){ NODES[n.id] = n; });
var RUNS = P.runs;

RUNS.forEach(function(r){
  r.end = r.steps[r.steps.length-1].t + r.dur[r.dur.length-1];
});
var TEND = Math.max.apply(null, RUNS.map(function(r){ return r.end; }));

// Every moment at which some run changes step or finishes. Runs have different
// lengths, so stepping through one run's steps would skip the others' events.
var EVENTS = (function(){
  var ts = [];
  RUNS.forEach(function(r){
    r.steps.forEach(function(s){ ts.push(s.t); });
    ts.push(r.end);
  });
  ts.sort(function(a,b){ return a-b; });
  return ts.filter(function(v,i){ return i===0 || v-ts[i-1] > 1e-9; });
})();

function css(v){ return getComputedStyle(document.body).getPropertyValue(v).trim(); }
function radius(k){ return k === "gpu" ? 21 : 16; }
function isSw(L){ return L.v.length > 0 || NODES[L.s].kind !== "gpu" || NODES[L.d].kind !== "gpu"; }

function waypoints(L){
  var pts = [[NODES[L.s].x, NODES[L.s].y]];
  L.v.forEach(function(w){ if (NODES[w]) pts.push([NODES[w].x, NODES[w].y]); });
  pts.push([NODES[L.d].x, NODES[L.d].y]);
  pts[0] = toward(pts[0], pts[1], radius(NODES[L.s].kind));
  var n = pts.length - 1;
  pts[n] = toward(pts[n], pts[n-1], radius(NODES[L.d].kind));
  return nudge(pts, L.s < L.d ? 4.5 : -4.5);
}
function toward(a,b,r){
  var dx=b[0]-a[0], dy=b[1]-a[1], d=Math.hypot(dx,dy)||1;
  return [a[0]+dx/d*r, a[1]+dy/d*r];
}
function nudge(pts, amt){
  var a=pts[0], b=pts[pts.length-1];
  var dx=b[0]-a[0], dy=b[1]-a[1], d=Math.hypot(dx,dy)||1;
  var nx=-dy/d*amt, ny=dx/d*amt;
  return pts.map(function(p){ return [p[0]+nx, p[1]+ny]; });
}
function arc(pts){
  var segs=[], total=0;
  for (var i=1;i<pts.length;i++){
    var L=Math.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1]);
    segs.push(L); total+=L;
  }
  return {segs:segs, total:total};
}
function along(pts, meta, f){
  var want = f * meta.total, acc = 0;
  for (var i=0;i<meta.segs.length;i++){
    if (acc + meta.segs[i] >= want){
      var u = meta.segs[i] ? (want-acc)/meta.segs[i] : 0;
      return [pts[i][0] + (pts[i+1][0]-pts[i][0])*u,
              pts[i][1] + (pts[i+1][1]-pts[i][1])*u];
    }
    acc += meta.segs[i];
  }
  return pts[pts.length-1];
}

function stepAt(run, t){
  if (t >= run.end - 1e-9) return -1;          // done: nothing is on the wire
  for (var i=run.steps.length-1;i>=0;i--){ if (t >= run.steps[i].t) return i; }
  return 0;
}

function Panel(run, idx){
  var el = document.createElement("div");
  el.className = "panel";
  el.innerHTML =
    '<div class="phead"><h2 class="ptitle">'+run.label+'</h2></div>'+
    '<dl class="stats">'+
      '<div class="stat"><dt>traffic</dt><dd>'+fmt(run.objective)+'</dd></div>'+
      '<div class="stat"><dt>latency</dt><dd>'+fmt(run.latency)+'<span class="u">s</span></dd></div>'+
      '<div class="stat"><dt>alg bw</dt><dd>'+fmt(run.bw,1)+'<span class="u">GB/s</span></dd></div>'+
      '<div class="stat"><dt>overhead</dt><dd>'+fmt(run.overhead*100,1)+'<span class="u">%</span></dd></div>'+
    '</dl>';
  var cv = document.createElement("canvas");
  el.appendChild(cv);
  var now = document.createElement("div");
  now.className = "now";
  el.appendChild(now);
  this.el=el; this.cv=cv; this.ctx=cv.getContext("2d"); this.run=run; this.now=now; this.idx=idx;
  this.cache={};
}
function fmt(v,d){ return (v==null) ? "--" : Number(v).toFixed(d==null?2:d); }

Panel.prototype.resize = function(){
  var w = this.cv.clientWidth || 360;
  var dpr = window.devicePixelRatio || 1;
  this.cv.width = Math.round(w*dpr);
  this.cv.height = Math.round(w*dpr);
  this.cv.style.height = w + "px";
  this.scale = w/SIZE*dpr;
};

Panel.prototype.geom = function(L){
  var k = L.s+">"+L.d+"|"+L.v.join("+");
  if (!this.cache[k]){
    var pts = waypoints(L);
    this.cache[k] = {pts:pts, meta:arc(pts)};
  }
  return this.cache[k];
};

Panel.prototype.draw = function(t, clock){
  var ctx=this.ctx, s=this.scale;
  ctx.setTransform(s,0,0,s,0,0);
  ctx.clearRect(0,0,SIZE,SIZE);

  var cLine=css("--line"), cRing=css("--ring"), cSw=css("--switch"),
      cGpu=css("--gpu-fill"), cGpuInk=css("--gpu-ink"), cMem=css("--mem"), cSurf=css("--surface");

  // idle topology
  ctx.strokeStyle=cLine; ctx.lineWidth=1.4;
  P.edges.forEach(function(e){
    var a=NODES[e[0]], b=NODES[e[1]];
    if(!a||!b) return;
    var p=toward([a.x,a.y],[b.x,b.y],radius(a.kind));
    var q=toward([b.x,b.y],[a.x,a.y],radius(b.kind));
    ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
  });

  var si = stepAt(this.run, t);
  var self = this;
  var step = si < 0 ? {links: []} : this.run.steps[si];

  // active links
  step.links.forEach(function(L){
    var g = self.geom(L), sw = isSw(L);
    ctx.strokeStyle = sw?cSw:cRing;
    ctx.lineWidth = 1.5 + 5.5*(L.kB/self.run.peak);
    ctx.lineCap="round"; ctx.lineJoin="round";
    ctx.globalAlpha=0.30;
    ctx.beginPath(); ctx.moveTo(g.pts[0][0],g.pts[0][1]);
    for(var i=1;i<g.pts.length;i++) ctx.lineTo(g.pts[i][0],g.pts[i][1]);
    ctx.stroke();
    ctx.globalAlpha=1;
  });

  // packets: constant speed, count proportional to volume
  var SPEED = 150; // canvas units per schedule-second
  step.links.forEach(function(L){
    var g = self.geom(L), sw = isSw(L);
    var n = Math.max(1, Math.round(7*(L.kB/self.run.peak)));
    var period = g.meta.total/SPEED;
    var base = (clock/period) % 1;
    ctx.fillStyle = sw?cSw:cRing;
    for(var i=0;i<n;i++){
      var f = (base + i/n) % 1;
      var p = along(g.pts, g.meta, f);
      ctx.beginPath(); ctx.arc(p[0],p[1], 2.6, 0, 6.2832); ctx.fill();
    }
  });

  // nodes
  P.nodes.forEach(function(nd){
    var r=radius(nd.kind);
    ctx.beginPath();
    if(nd.kind==="gpu"){ ctx.fillStyle=cGpu; ctx.arc(nd.x,nd.y,r,0,6.2832); ctx.fill(); }
    else {
      ctx.fillStyle=cSurf;
      ctx.strokeStyle = nd.kind==="mem"?cMem:(nd.fan?cSw:cLine);
      ctx.lineWidth = nd.fan?2.5:1.6;
      if(ctx.roundRect){ ctx.roundRect(nd.x-r,nd.y-r,r*2,r*2,6); }
      else { ctx.rect(nd.x-r,nd.y-r,r*2,r*2); }
      ctx.fill(); ctx.stroke();
      if(nd.fan){ // inner ring marks copy/reduce capability
        ctx.beginPath(); ctx.strokeStyle=cSw; ctx.lineWidth=1;
        if(ctx.roundRect){ ctx.roundRect(nd.x-r+4,nd.y-r+4,(r-4)*2,(r-4)*2,4); }
        else { ctx.rect(nd.x-r+4,nd.y-r+4,(r-4)*2,(r-4)*2); }
        ctx.stroke();
      }
    }
    ctx.fillStyle = nd.kind==="gpu"?cGpuInk:(nd.kind==="mem"?cMem:(nd.fan?cSw:css("--muted")));
    ctx.font="600 11px "+css("--mono");
    ctx.textAlign="center"; ctx.textBaseline="middle";
    ctx.fillText(nd.id, nd.x, nd.y);
  });

  if (si < 0) {
    this.now.innerHTML = '<b class="done">finished at '+this.run.end.toFixed(3)+
      ' s</b> &middot; idle<div class="row"></div>';
    return;
  }
  var tags = step.links.slice(0,8).map(function(L){
    return '<span class="tag '+(isSw(L)?"sw":"rg")+'">'+L.s+(L.v.length?"⇢":"→")+L.d+
           " · "+Math.round(L.kB/1000)+"MB</span>";
  }).join("");
  this.now.innerHTML = "<b>step "+si+"</b> · "+step.links.length+" active link"+
    (step.links.length===1?"":"s")+'<div class="row">'+tags+"</div>";
};

var panels = RUNS.map(function(r,i){ return new Panel(r,i); });
var host = document.getElementById("panels");
panels.forEach(function(p){ host.appendChild(p.el); p.resize(); });
window.addEventListener("resize", function(){ panels.forEach(function(p){ p.resize(); }); paint(); });

var scrub=document.getElementById("scrub"), sn=document.getElementById("sn"),
    tt=document.getElementById("tt"), playBtn=document.getElementById("play"),
    speedSel=document.getElementById("speed");
scrub.max=String(TEND);

var T=0, playing=false, last=null, clock=0;

function paint(){
  panels.forEach(function(p){ p.draw(T, clock); });
  scrub.value=String(T);
  tt.textContent=T.toFixed(3);
  var ei = 0;
  for (var i=EVENTS.length-1;i>=0;i--){ if (T >= EVENTS[i]-1e-9){ ei=i; break; } }
  sn.textContent = ei;
  document.getElementById("st").textContent = " / "+(EVENTS.length-1);
  var live = RUNS.filter(function(r){ return T < r.end-1e-9; }).length;
  document.getElementById("live").textContent = live+"/"+RUNS.length;
}

function frame(ts){
  if(!playing) return;
  if(last==null) last=ts;
  var dt=(ts-last)/1000; last=ts;
  var mult=parseFloat(speedSel.value);
  T += dt*mult;
  clock += dt*mult;
  if(T>=TEND){ T=TEND; playing=false; playBtn.textContent="Play"; }
  paint();
  if(playing) requestAnimationFrame(frame);
}

function play(){
  if(T>=TEND) { T=0; clock=0; }
  playing=true; last=null; playBtn.textContent="Pause";
  requestAnimationFrame(frame);
}
function pause(){ playing=false; playBtn.textContent="Play"; }

playBtn.addEventListener("click", function(){ playing?pause():play(); });
scrub.addEventListener("input", function(){ pause(); T=parseFloat(scrub.value); clock=T; paint(); });

function jump(dir){
  pause();
  var i = 0;
  for (var k=EVENTS.length-1;k>=0;k--){ if (T >= EVENTS[k]-1e-9){ i=k; break; } }
  i = Math.max(0, Math.min(EVENTS.length-1, i+dir));
  T = EVENTS[i]; clock = T; paint();
}
document.getElementById("prev").addEventListener("click", function(){ jump(-1); });
document.getElementById("next").addEventListener("click", function(){ jump(1); });

// idle animation so the packets move even when paused at a step
(function idle(ts){
  if(!playing){ clock += 1/60; paint(); }
  requestAnimationFrame(idle);
})();

document.getElementById("foot").textContent = __FOOT__;
paint();
})();
</script>
"""


def render_html(payload: dict, title: str, heading: str, eyebrow: str,
                foot: str) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("__TITLE__", title)
        .replace("__HEADING__", heading)
        .replace("__EYEBROW__", eyebrow)
        .replace("__FOOT__", json.dumps(foot))
        .replace("__DATA__", data)
    )


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Render OptCCL schedule JSON as an interactive HTML page."
    )
    ap.add_argument("schedules", nargs="+", help="schedule JSON from --write-schedule")
    ap.add_argument("--topology", required=True,
                    help="topology spec or built-in name used for the solve")
    ap.add_argument("--num-nodes", type=int, default=1)
    ap.add_argument("--label", action="append", default=[],
                    help="panel label (repeat, one per schedule)")
    ap.add_argument("--layout", default="shell",
                    choices=["shell", "spring", "circular"])
    ap.add_argument("--size", type=int, default=640, help="canvas units")
    ap.add_argument("--title", default="Schedule Scope")
    ap.add_argument("--heading", default="What the solver actually does")
    ap.add_argument("-o", "--output", default="schedule_viz.html")
    args = ap.parse_args(argv)

    for path in args.schedules:
        if not os.path.isfile(path):
            raise SystemExit(f"schedule not found: {path}")

    builtins = {"doe", "doe_pcie", "doe_both", "a100", "a100_pcie", "a100_both"}
    if args.topology not in builtins and not os.path.isfile(args.topology):
        raise SystemExit(
            f"topology not found: {args.topology}\n"
            f"Pass a path to a topology spec, or one of: {', '.join(sorted(builtins))}"
        )

    cfg = Config()
    top = load_or_build_topology(args.topology, args.num_nodes, cfg)

    runs = []
    for i, path in enumerate(args.schedules):
        r = load_schedule(path)
        r["label"] = (args.label[i] if i < len(args.label)
                      else os.path.splitext(os.path.basename(path))[0])
        runs.append(r)

    pos = layout(top, args.layout, args.size)
    payload = build_payload(runs, top, pos, args.size)

    eyebrow = f"{len(top.gpus)} GPUs · {len(top.components)} components · " \
              f"{len(runs)} run{'s' if len(runs) != 1 else ''}"
    if len(runs) == 2 and all(r["objective"] for r in runs):
        a, b = runs[0]["objective"], runs[1]["objective"]
        foot = (f"{runs[1]['label']} moves {(1 - b / a) * 100:.1f}% "
                f"less data than {runs[0]['label']} "
                f"({b:.2f} vs {a:.2f} link-hops).")
    else:
        foot = f"Generated from {', '.join(os.path.basename(p) for p in args.schedules)}."

    html = render_html(payload, args.title, args.heading, eyebrow, foot)
    with open(args.output, "w") as fh:
        fh.write(html)
    print(f"wrote {args.output}  ({len(html) / 1024:.1f} KB, "
          f"{len(payload['nodes'])} nodes, {len(payload['edges'])} edges)")


if __name__ == "__main__":
    main()
