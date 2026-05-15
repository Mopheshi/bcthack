// js/renderers.js
// Pure rendering functions: data -> HTML string -> innerHTML.

import { escHtml, formatLatency } from './utils.js';

export function renderTaskA(data, container, latencyMs) {
  const ps    = data.predicted_stars || 0;
  const stars = '★'.repeat(Math.round(ps)) + '☆'.repeat(5 - Math.round(ps));
  const conf  = Math.round((data.rating_confidence || 0) * 100);
  const p     = data.persona_summary || {};

  const nb   = p.nigerian ? '<span class="tag tag-gold">🇳🇬 Nigerian Mode</span>' : '';
  const cb   = p.is_cold
    ? '<span class="tag tag-red">Cold Start</span>'
    : '<span class="tag tag-teal">Warm User</span>';
  const bb   = p.rating_bias ? `<span class="tag tag-neutral">${escHtml(p.rating_bias)}</span>` : '';
  const sb   = p.style ? `<span class="tag tag-neutral">${escHtml(p.style)}</span>` : '';
  const cats = (p.top_cats || []).slice(0, 3)
    .map(c => `<span class="tag tag-neutral">${escHtml(c)}</span>`)
    .join('');

  container.innerHTML = `
    <div class="stars-display">
      <div class="star-val">${ps}</div>
      <div class="stars-glyphs">${stars}</div>
      <div class="confidence-bar-wrap">
        <div class="confidence-label">Confidence ${conf}%</div>
        <div class="confidence-bar"><div class="confidence-fill" style="width:${conf}%"></div></div>
      </div>
    </div>
    <div class="review-text">${escHtml(data.review_text || '')}</div>
    <div class="tag-row">${nb}${cb}${bb}${sb}${cats}</div>
    ${p.review_count !== undefined
      ? `<div class="meta-line">Based on ${p.review_count} historical reviews · ${formatLatency(latencyMs)}</div>`
      : `<div class="meta-line">Completed in ${formatLatency(latencyMs)}</div>`}`;
}

export function renderTaskB(data, container, latencyMs) {
  const p    = data.persona_summary || {};
  const recs = data.recommendations || [];

  const nb = p.nigerian ? '<span class="tag tag-gold">🇳🇬 Nigerian Mode</span>' : '';
  const cb = p.is_cold
    ? '<span class="tag tag-red">Cold Start</span>'
    : '<span class="tag tag-teal">Warm User</span>';

  const items = recs.length
    ? recs.map((r, i) => `
        <div class="rec-item">
          <div class="rec-rank">#${i + 1}</div>
          <div class="rec-body">
            <div class="rec-name">${escHtml(r.name || r.business_id)}</div>
            <div class="rec-meta">${escHtml([r.categories, r.city].filter(Boolean).join(' · ').slice(0, 80))}</div>
            <div class="rec-reason">${escHtml(r.reason || '')}</div>
          </div>
          <div class="rec-score">
            ${r.stars ? r.stars + '★' : ''}
            <span class="rec-score-pct">${r.score ? (r.score * 100).toFixed(0) + '%' : ''}</span>
          </div>
        </div>`).join('')
    : '<div class="meta-line">No recommendations returned.</div>';

  container.innerHTML = `
    ${data.search_intent ? `<div class="intent-chip">⟳ ${escHtml(data.search_intent)}</div>` : ''}
    <div class="tag-row" style="margin-bottom:14px">${nb}${cb}</div>
    <div class="rec-list">${items}</div>
    <div class="meta-line">Completed in ${formatLatency(latencyMs)}</div>`;
}

export function renderError(err, container, apiUrl) {
  const msg = err.timeout ? err.message
            : err.status  ? `${err.message}`
            : `${err.message}`;
  container.innerHTML = `
    <div class="error-box">
      ⚠ ${escHtml(msg)}<br><br>
      Endpoint: ${escHtml(apiUrl)}
    </div>`;
}
