/* Shared canvas plotter for the workbench pages. No dependencies.

   plot(canvas, series, opts)
     series  [{ x, y, label, color, width, dash, dots, kind, r }]
             kind: 'line' (default) | 'bars' (histogram: x = equal-width bin centres) | 'dots'
             color: any CSS colour, or a token name such as '--s2'. Tokens resolve at draw time,
             so a theme switch never leaves light-mode colours on a dark surface.
     opts    { height, xlabel, ylabel, yFloor, yMin0, xlog, ylog, xRange, yRange, labels, empty,
               hlines: [{ y, color, dash, label }], vlines: [{ x, color, dash, label }],
               bands:  [{ x0, x1, color, label }],  hbands: [{ y0, y1, color, label }] }

   Hover: a crosshair snaps to the nearest x of the first series and one tooltip lists every
   series there. The canvas takes keyboard focus and the arrow keys step the same positions
   (shift for bigger steps). A <details class="data" data-for="<canvas id>"> beside the canvas
   shows the plotted numbers as a table, so no value is reachable only by hovering. */

function css(v) { return getComputedStyle(document.body).getPropertyValue(v).trim(); }

/* Validated categorical slots, then muted. A live array: refreshed when the theme changes. */
const SERIES = [];
function refreshSeries() {
  SERIES.length = 0;
  ['--s1', '--s2', '--s3', '--s4', '--muted'].forEach(t => SERIES.push(css(t)));
}
refreshSeries();
(function watchTheme() {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const changed = () => { refreshSeries(); window.dispatchEvent(new Event('themechange')); };
  if (mq.addEventListener) mq.addEventListener('change', changed); else mq.addListener(changed);
})();

function colour(c, i) {
  if (!c) return SERIES[(i || 0) % SERIES.length];
  return c.startsWith('--') ? css(c) : c;
}

/* Format a tick against the axis RANGE, not its own magnitude: two neighbouring ticks
   must never round to the same string (0.001 and 0.001 for a 7e-4 span). */
function fmt(v, range) {
  const a = Math.abs(v);
  if (a < (range || 1) * 1e-9) return '0';
  const span = range || a || 1;
  if (span >= 1e4 || span < 1e-3 || a >= 1e5)
    return v.toExponential(Math.abs(v) >= 1 ? 2 : 1).replace('e+', 'e');
  const decimals = Math.max(0, Math.min(6, Math.ceil(-Math.log10(span / 4)) + 1));
  return v.toFixed(decimals);
}

/* A single value at readable precision: tooltips and tables. */
function fmtSig(v) {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(3).replace('e+', 'e');
  return String(+v.toPrecision(4));
}

function fmtDecade(e) {
  return e >= -2 && e <= 4 ? String(+(10 ** e).toPrecision(1)) : '1e' + e;
}

/* Bin width for a bar series: explicit binWidth wins, then the spacing of the first two bins.
   A single-bin histogram has no spacing, and falling back to 1 data unit drew it 1/dt wide. */
function barStep(s, span) {
  if (s.binWidth) return Math.abs(s.binWidth);
  if (s.x.length > 1) return Math.abs(s.x[1] - s.x[0]);
  return Math.abs(span) / 20 || 1;
}

function arrMin(a) { let m = Infinity; for (const v of a) if (v < m) m = v; return m; }
function arrMax(a) { let m = -Infinity; for (const v of a) if (v > m) m = v; return m; }

function plot(canvas, series, opts) {
  opts = opts || {};
  attachHover(canvas);
  if (canvas._hideTip) canvas._hideTip();   // a tooltip left open would keep the old run's values
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth || 620, H = opts.height || 360;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.height = H + 'px';
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const L = 76, R = 14, T = 12, B = 40;   // L leaves room for tick labels AND the rotated title
  const muted = css('--muted'), rule = css('--rule');
  const xlog = !!opts.xlog, ylog = !!opts.ylog;
  const tx = xlog ? Math.log10 : v => v, ty = ylog ? Math.log10 : v => v;
  const okx = v => v !== null && v !== undefined && isFinite(v) && (!xlog || v > 0);
  const oky = v => v !== null && v !== undefined && isFinite(v) && (!ylog || v > 0);

  const live = (series || [])
    .map((s, i) => s && s.x && s.x.length
      ? Object.assign({}, s, { _c: colour(s.color, i),
                               label: s.label || (opts.labels && opts.labels[i]) || `series ${i + 1}` })
      : null)
    .filter(Boolean);
  if (!live.length) {
    canvas._geo = null;
    g.fillStyle = muted; g.font = '12px system-ui';
    g.fillText(opts.empty || 'run a propagation to see this', L, H / 2);
    if (canvas._table && canvas._table.open) fillTable(canvas._table, canvas);
    return;
  }

  const xs = [], ys = [];
  live.forEach(s => {
    for (let k = 0; k < s.x.length; k++)
      if (okx(s.x[k]) && oky(s.y[k])) { xs.push(tx(s.x[k])); ys.push(ty(s.y[k])); }
  });
  if (!xs.length) {
    canvas._geo = null;
    g.fillStyle = muted; g.font = '12px system-ui';
    g.fillText(opts.empty || 'nothing to plot on this scale', L, H / 2);
    if (canvas._table && canvas._table.open) fillTable(canvas._table, canvas);
    return;
  }
  let x0 = arrMin(xs), x1 = arrMax(xs), y0 = arrMin(ys), y1 = arrMax(ys);
  const hasBars = live.some(s => s.kind === 'bars');
  live.filter(s => s.kind === 'bars').forEach(s => {
    const half = barStep(s, x1 - x0) / 2;
    x0 = Math.min(x0, tx(s.x[0] - half)); x1 = Math.max(x1, tx(s.x[s.x.length - 1] + half));
  });
  (opts.vlines || []).forEach(v => { if (okx(v.x)) { x0 = Math.min(x0, tx(v.x)); x1 = Math.max(x1, tx(v.x)); } });
  (opts.bands || []).forEach(b => [b.x0, b.x1].forEach(v => {
    if (okx(v)) { x0 = Math.min(x0, tx(v)); x1 = Math.max(x1, tx(v)); } }));
  (opts.hlines || []).forEach(h => { if (oky(h.y)) { y0 = Math.min(y0, ty(h.y)); y1 = Math.max(y1, ty(h.y)); } });
  (opts.hbands || []).forEach(b => [b.y0, b.y1].forEach(v => {
    if (oky(v)) { y0 = Math.min(y0, ty(v)); y1 = Math.max(y1, ty(v)); } }));
  const allNonNeg = ys.every(v => ylog || v >= 0);
  if (hasBars && !ylog) y0 = Math.min(y0, 0);
  if (opts.yFloor !== undefined) y0 = Math.max(y0, ty(opts.yFloor));
  // A flat trace (CW power, a constant spectrum) has zero range: pad relative to the
  // value itself, never by a fixed +-1, which would put a power axis below zero.
  if (!isFinite(x0) || !isFinite(x1) || x0 === x1) {
    const dx = Math.abs(x0) * 0.05 || 1; x0 -= dx; x1 += dx;
  }
  if (!isFinite(y0) || !isFinite(y1) || y0 === y1) {
    const dy = Math.abs(y0) * 0.25 || 1; y0 -= dy; y1 += dy;
  }
  const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  if ((opts.yMin0 || (hasBars && allNonNeg)) && !ylog && y0 < 0) y0 = 0;   // counts and power cannot be negative
  if (opts.xRange) { x0 = tx(opts.xRange[0]); x1 = tx(opts.xRange[1]); }
  if (opts.yRange) { y0 = ty(opts.yRange[0]); y1 = ty(opts.yRange[1]); }

  const X = t => L + (t - x0) / (x1 - x0) * (W - L - R);
  const Y = t => H - B - (t - y0) / (y1 - y0) * (H - T - B);
  const sx = v => X(tx(v)), sy = v => Y(Math.max(ty(v), y0));

  // bands first: they sit behind the grid and the data
  (opts.bands || []).forEach(b => {
    if (!okx(b.x0) || !okx(b.x1)) return;
    g.globalAlpha = b.alpha || 0.13; g.fillStyle = colour(b.color || '--s2');
    const a = Math.max(L, sx(Math.min(b.x0, b.x1))), z = Math.min(W - R, sx(Math.max(b.x0, b.x1)));
    if (z > a) g.fillRect(a, T, z - a, H - T - B);
    g.globalAlpha = 1;
  });
  (opts.hbands || []).forEach(b => {
    if (!oky(b.y0) || !oky(b.y1)) return;
    g.globalAlpha = b.alpha || 0.13; g.fillStyle = colour(b.color || '--muted');
    const top = Math.max(T, sy(Math.max(b.y0, b.y1))), bot = Math.min(H - B, sy(Math.min(b.y0, b.y1)));
    if (bot > top) g.fillRect(L, top, W - L - R, bot - top);
    g.globalAlpha = 1;
  });

  const ticks = (lo, hi, log) => {
    if (log && hi - lo >= 1) {
      const a = Math.ceil(lo - 1e-9), b = Math.floor(hi + 1e-9), step = Math.max(1, Math.ceil((b - a) / 6));
      const out = [];
      for (let e = a; e <= b; e += step) out.push({ t: e, label: fmtDecade(e) });
      return out;
    }
    const out = [];
    for (let i = 0; i <= 4; i++) {
      const t = lo + (hi - lo) * i / 4;
      out.push({ t, label: log ? fmtSig(10 ** t) : fmt(t, hi - lo) });
    }
    return out;
  };
  g.strokeStyle = rule; g.lineWidth = 1; g.setLineDash([]); g.fillStyle = muted;
  g.font = '10px ui-monospace, monospace';
  ticks(y0, y1, ylog).forEach(tk => {
    const yy = Math.round(Y(tk.t)) + 0.5;
    g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke();
    g.textAlign = 'right'; g.textBaseline = 'middle';
    g.fillText(tk.label, L - 7, yy);
  });
  ticks(x0, x1, xlog).forEach(tk => {
    const xx = Math.round(X(tk.t)) + 0.5;
    g.textAlign = 'center'; g.textBaseline = 'top';
    g.fillText(tk.label, xx, H - B + 7);
  });
  g.strokeStyle = rule;
  g.beginPath(); g.moveTo(L, T); g.lineTo(L, H - B); g.lineTo(W - R, H - B); g.stroke();

  g.fillStyle = muted; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
  g.fillText(opts.xlabel || '', (L + W - R) / 2, H - 10);
  g.save(); g.translate(12, (T + H - B) / 2); g.rotate(-Math.PI / 2);
  g.textBaseline = 'top'; g.fillText(opts.ylabel || '', 0, 0); g.restore();

  g.save();
  // 4px of slack so a marker sitting on the first or last x is not sliced in half
  g.beginPath(); g.rect(L - 4, T - 2, W - L - R + 8, H - T - B + 2); g.clip();
  live.forEach(s => {
    const col = s._c;
    if (s.kind === 'bars') {
      const n = s.x.length, step = barStep(s, x1 - x0);
      const px = Math.abs(sx(s.x[0] + step) - sx(s.x[0]));
      const base = Y(ylog ? y0 : Math.max(0, y0));
      if (px < 3) {                 // too narrow for gaps: a stepped outline over a light wash
        g.beginPath(); g.moveTo(sx(s.x[0] - step / 2), base);
        for (let k = 0; k < n; k++) {
          const top = oky(s.y[k]) ? sy(s.y[k]) : base;
          g.lineTo(sx(s.x[k] - step / 2), top); g.lineTo(sx(s.x[k] + step / 2), top);
        }
        g.lineTo(sx(s.x[n - 1] + step / 2), base); g.closePath();
        g.globalAlpha = 0.18; g.fillStyle = col; g.fill(); g.globalAlpha = 1;
        g.strokeStyle = col; g.lineWidth = 1.2; g.setLineDash([]); g.stroke();
      } else {                      // separate bars with a surface gap, square at the baseline
        const gap = px >= 8 ? 2 : 1;
        g.fillStyle = col;
        for (let k = 0; k < n; k++) {
          if (!oky(s.y[k])) continue;
          const a = sx(s.x[k] - step / 2) + gap / 2, z = sx(s.x[k] + step / 2) - gap / 2;
          const top = sy(s.y[k]), h = base - top;
          if (h <= 0 || z <= a) continue;
          const r = Math.min(3, (z - a) / 3, h);
          g.beginPath();
          g.moveTo(a, base); g.lineTo(a, top + r); g.quadraticCurveTo(a, top, a + r, top);
          g.lineTo(z - r, top); g.quadraticCurveTo(z, top, z, top + r); g.lineTo(z, base);
          g.closePath(); g.fill();
        }
      }
      return;
    }
    g.strokeStyle = col; g.lineWidth = s.width || 1.6;
    g.lineJoin = 'round'; g.lineCap = 'round';
    g.setLineDash(s.dash || []);
    if (s.kind !== 'dots') {
      g.beginPath();
      let started = false;
      for (let k = 0; k < s.x.length; k++) {
        if (!okx(s.x[k]) || !oky(s.y[k])) { started = false; continue; }
        const px = sx(s.x[k]), py = sy(s.y[k]);
        if (!started) { g.moveTo(px, py); started = true; } else g.lineTo(px, py);
      }
      g.stroke();
    }
    g.setLineDash([]);
    if (s.dots || s.kind === 'dots') {
      const r = s.r || (s.kind === 'dots' ? 1.8 : 2.6);
      g.fillStyle = col;
      for (let k = 0; k < s.x.length; k++) {
        if (!okx(s.x[k]) || !oky(s.y[k])) continue;
        g.beginPath(); g.arc(sx(s.x[k]), sy(s.y[k]), r, 0, 7); g.fill();
      }
    }
  });
  g.restore();

  // reference lines on top of the data; their labels wear text ink, never the line colour
  g.font = '10px ui-monospace, monospace';
  (opts.hlines || []).forEach(h => {
    if (!oky(h.y) || ty(h.y) < y0 || ty(h.y) > y1) return;
    const yy = Math.round(sy(h.y)) + 0.5;
    g.strokeStyle = colour(h.color || '--muted'); g.lineWidth = h.width || 1.2;
    g.setLineDash(h.dash || [5, 4]);
    g.beginPath(); g.moveTo(L, yy); g.lineTo(W - R, yy); g.stroke(); g.setLineDash([]);
    if (h.label) {
      g.fillStyle = muted; g.textAlign = 'right'; g.textBaseline = 'bottom';
      g.fillText(h.label, W - R - 3, yy - 3);
    }
  });
  (opts.vlines || []).forEach(v => {
    if (!okx(v.x) || tx(v.x) < x0 || tx(v.x) > x1) return;
    const xx = Math.round(sx(v.x)) + 0.5;
    g.strokeStyle = colour(v.color || '--muted'); g.lineWidth = v.width || 1.2;
    g.setLineDash(v.dash || []);
    g.beginPath(); g.moveTo(xx, T); g.lineTo(xx, H - B); g.stroke(); g.setLineDash([]);
    if (v.label) {
      g.fillStyle = muted; g.textBaseline = 'top';
      const right = xx > (L + W - R) / 2;
      g.textAlign = right ? 'right' : 'left';
      g.fillText(v.label, xx + (right ? -4 : 4), T + 2);
    }
  });

  canvas._geo = { W, H, L, R, T, B, x0, x1, y0, y1, xlog, ylog, live, opts };
  if (canvas._table && canvas._table.open) fillTable(canvas._table, canvas);
}

function legend(el, items, colors) {
  el.textContent = '';
  (items || []).forEach((it, i) => {
    const o = typeof it === 'string' ? { label: it } : it;
    const span = document.createElement('span'), key = document.createElement('i');
    const c = colour(o.color || (colors && colors[i]), i);
    if (o.kind === 'bars') key.className = 'bar';
    if (o.kind === 'band') key.className = 'band';
    if (o.kind === 'dash') { key.className = 'dash'; key.style.borderTopColor = c; }
    else key.style.background = c;
    span.append(key, document.createTextNode(o.label));
    el.appendChild(span);
  });
}

/* ---------------- hover, keyboard and table view ---------------- */
function nearest(xs, v) {
  const n = xs.length;
  if (!n) return null;
  const asc = xs[0] <= xs[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if ((xs[mid] < v) === asc) lo = mid; else hi = mid;
  }
  return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
}

function attachHover(canvas) {
  if (canvas._hoverReady) return;
  canvas._hoverReady = true;
  let wrap = canvas.parentElement;
  if (!wrap.classList.contains('plotwrap')) {
    wrap = document.createElement('div'); wrap.className = 'plotwrap';
    canvas.parentNode.insertBefore(wrap, canvas); wrap.appendChild(canvas);
  }
  const hair = document.createElement('div'); hair.className = 'hair'; hair.hidden = true;
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true;
  wrap.append(hair, tip);
  canvas.tabIndex = 0;
  let idx = null;
  const hide = () => { hair.hidden = true; tip.hidden = true; idx = null; };
  canvas._hideTip = hide;
  const pxOf = (G, v) => G.L + ((G.xlog ? Math.log10(v) : v) - G.x0) / (G.x1 - G.x0) * (G.W - G.L - G.R);

  function show(k) {
    const G = canvas._geo;
    if (!G || k === null) return hide();
    const lead = G.live[0];
    idx = k;
    const xv = lead.x[k], px = pxOf(G, xv);
    hair.style.left = Math.round(px) + 'px';
    hair.style.height = (G.H - G.T - G.B) + 'px';
    hair.hidden = false;
    tip.textContent = '';
    const head = document.createElement('div'); head.className = 'tx';
    head.textContent = `${G.opts.xlabel || 'x'}: ${fmtSig(xv)}`;
    tip.appendChild(head);
    const addRow = (value, label, c, dashed) => {
      const row = document.createElement('div'); row.className = 'tr';
      const key = document.createElement('i');
      if (dashed) { key.style.borderTop = `2px dashed ${c}`; key.style.height = '0'; } else key.style.background = c;
      const b = document.createElement('b'); b.textContent = fmtSig(value);
      const span = document.createElement('span'); span.textContent = label;
      row.append(key, b, span); tip.appendChild(row);
    };
    G.live.forEach(s => {
      const j = s === lead ? k : nearest(s.x, xv);
      if (j === null || !isFinite(s.y[j]) || s.y[j] === null) return;
      if (s !== lead && Math.abs(pxOf(G, s.x[j]) - px) > 12) return;   // no value from a far-off point
      addRow(s.y[j], s.label, s._c, false);
    });
    (G.opts.hlines || []).forEach(h => { if (h.label && isFinite(h.y)) addRow(h.y, h.label, colour(h.color || '--muted'), true); });
    tip.hidden = false;
    const tw = tip.offsetWidth;
    let left = px + 12;
    if (left + tw > G.W - 4) left = px - 12 - tw;
    tip.style.left = Math.max(0, left) + 'px';
  }

  canvas.addEventListener('pointermove', e => {
    const G = canvas._geo;
    if (!G) return hide();
    const px = e.clientX - canvas.getBoundingClientRect().left;
    if (px < G.L || px > G.W - G.R) return hide();
    const t = G.x0 + (px - G.L) / (G.W - G.L - G.R) * (G.x1 - G.x0);
    show(nearest(G.live[0].x, G.xlog ? 10 ** t : t));
  });
  canvas.addEventListener('pointerleave', hide);
  canvas.addEventListener('blur', hide);
  canvas.addEventListener('keydown', e => {
    const G = canvas._geo;
    if (!G) return;
    const n = G.live[0].x.length;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      e.preventDefault();
      const step = e.shiftKey ? Math.max(1, Math.round(n / 20)) : 1;
      const next = (idx === null ? (e.key === 'ArrowRight' ? -1 : n) : idx) + (e.key === 'ArrowRight' ? step : -step);
      show(Math.max(0, Math.min(n - 1, next)));
    }
    if (e.key === 'Escape') hide();
  });

  if (canvas.id) {
    const d = document.querySelector(`details.data[data-for="${canvas.id}"]`);
    if (d) {
      canvas._table = d;
      d.addEventListener('toggle', () => { if (d.open) fillTable(d, canvas); });
    }
  }
}

function fillTable(d, canvas) {
  let wrap = d.querySelector('.tablewrap');
  if (!wrap) { wrap = document.createElement('div'); wrap.className = 'tablewrap'; d.appendChild(wrap); }
  wrap.textContent = '';
  const G = canvas._geo;
  if (!G) { wrap.textContent = 'Nothing plotted yet.'; return; }
  const same = (a, b) => a === b || (a.length === b.length && a.every((v, i) => v === b[i]));
  const groups = [];
  G.live.forEach(s => {
    const grp = groups.find(gr => same(gr.x, s.x));
    if (grp) grp.cols.push(s); else groups.push({ x: s.x, cols: [s] });
  });
  groups.forEach(grp => {
    const table = document.createElement('table'), head = document.createElement('tr');
    [G.opts.xlabel || 'x', ...grp.cols.map(s => s.label)].forEach(text => {
      const th = document.createElement('th'); th.textContent = text; head.appendChild(th);
    });
    const thead = document.createElement('thead'); thead.appendChild(head); table.appendChild(thead);
    const tbody = document.createElement('tbody');
    for (let k = 0; k < Math.min(grp.x.length, 2000); k++) {
      const tr = document.createElement('tr');
      [grp.x[k], ...grp.cols.map(s => s.y[k])].forEach(v => {
        const td = document.createElement('td'); td.textContent = fmtSig(v); tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody); wrap.appendChild(table);
  });
}
