const API_A = 'http://localhost:8001';
const API_B = 'http://localhost:8002';

let selectedStars = 4;
let conversationHistory = [];

function switchTab(t) {
    document.querySelectorAll('.tab').forEach((el, i) => {
        el.classList.toggle('active', (i === 0 && t === 'a') || (i === 1 && t === 'b'));
    });
    document.getElementById('panel-a').classList.toggle('active', t === 'a');
    document.getElementById('panel-b').classList.toggle('active', t === 'b');
}

function setStars(n) {
    selectedStars = n;
    document.querySelectorAll('.star-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i + 1 === n);
    });
}

function addTurn() {
    const role = document.getElementById('b-turn-role').value;
    const text = document.getElementById('b-turn-text').value.trim();
    if (!text) return;
    conversationHistory.push({ role, content: text });
    document.getElementById('b-turn-text').value = '';
    renderConvo();
}

function clearTurns() {
    conversationHistory = [];
    renderConvo();
}

function renderConvo() {
    const el = document.getElementById('b-convo-display');
    if (!conversationHistory.length) {
        el.innerHTML = '<div style="color:var(--text-muted);font-size:11px;font-family:var(--mono)">No turns yet</div>';
        return;
    }
    el.innerHTML = conversationHistory.map(t =>
        `<div class="convo-turn"><span class="convo-role ${t.role}">${t.role}</span><span style="color:var(--text-dark)">${h(t.content)}</span></div>`
    ).join('');
    el.scrollTop = el.scrollHeight;
}

async function runSimulate() {
    const btn = document.getElementById('a-btn');
    const result = document.getElementById('a-result');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>SIMULATING...';
    result.innerHTML = '';

    const payload = {
        user_id: document.getElementById('a-uid').value.trim() || 'cold_user',
        business_id: document.getElementById('a-bid').value.trim() || 'biz_001',
        product_details: {
            name: document.getElementById('a-name').value.trim(),
            categories: document.getElementById('a-cats').value.trim(),
            city: document.getElementById('a-city').value.trim(),
            state: document.getElementById('a-state').value.trim(),
            stars: selectedStars,
            review_count: 88
        }
    };

    try {
        const res = await fetch(`${API_A}/simulate-review`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        renderA(await res.json(), result);
    } catch (e) {
        result.innerHTML = `<div class="error-box">⚠ ${h(e.message)}<br><br>Ensure Task A is running at ${API_A}</div>`;
    }

    btn.disabled = false;
    btn.innerHTML = '▶ SIMULATE REVIEW';
}

function renderA(data, el) {
    const ps = data.predicted_stars || 0;
    const stars = '★'.repeat(Math.round(ps)) + '☆'.repeat(5 - Math.round(ps));
    const conf = Math.round((data.rating_confidence || 0) * 100);
    const p = data.persona_summary || {};

    const nb = p.nigerian ? '<span class="tag tag-gold">🇳🇬 Nigerian Mode</span>' : '';
    const cb = p.is_cold ? '<span class="tag tag-red">Cold Start</span>' : '<span class="tag tag-teal">Warm User</span>';
    const bb = p.rating_bias ? `<span class="tag tag-neutral">${p.rating_bias}</span>` : '';
    const sb = p.style ? `<span class="tag tag-neutral">${p.style}</span>` : '';
    const cats = (p.top_cats || []).slice(0, 3).map(c => `<span class="tag tag-neutral">${c}</span>`).join('');

    el.innerHTML = `
        <div class="stars-display">
            <div class="star-val">${ps}</div>
            <div style="color:var(--gold);font-size:18px;letter-spacing:2px">${stars}</div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label">Confidence ${conf}%</div>
                <div class="confidence-bar"><div class="confidence-fill" style="width:${conf}%"></div></div>
            </div>
        </div>
        <div class="review-text">${h(data.review_text || '')}</div>
        <div class="tag-row">${nb}${cb}${bb}${sb}${cats}</div>
        ${p.review_count !== undefined ? `<div style="color:var(--text-muted);font-size:11px;margin-top:12px;font-family:var(--mono)">Based on ${p.review_count} historical reviews</div>` : ''}
    `;
}

async function runRecommend() {
    const btn = document.getElementById('b-btn');
    const result = document.getElementById('b-result');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>REASONING...';
    result.innerHTML = '';

    const payload = {
        user_id: document.getElementById('b-uid').value.trim() || 'cold_user',
        context: document.getElementById('b-ctx').value.trim(),
        conversation_history: conversationHistory,
        top_k: parseInt(document.getElementById('b-topk').value)
    };

    try {
        const res = await fetch(`${API_B}/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
        renderB(await res.json(), result);
    } catch (e) {
        result.innerHTML = `<div class="error-box">⚠ ${h(e.message)}<br><br>Ensure Task B is running at ${API_B}</div>`;
    }

    btn.disabled = false;
    btn.innerHTML = '▶ GET RECOMMENDATIONS';
}

function renderB(data, el) {
    const p = data.persona_summary || {};
    const recs = data.recommendations || [];

    const nb = p.nigerian ? '<span class="tag tag-gold">🇳🇬 Nigerian Mode</span>' : '';
    const cb = data.cold_start ? '<span class="tag tag-red">Cold Start</span>' : '<span class="tag tag-teal">Warm User</span>';

    const items = recs.length ? recs.map((r, i) => `
        <div class="rec-item">
            <div class="rec-rank">#${i + 1}</div>
            <div class="rec-body">
                <div class="rec-name">${h(r.name || r.business_id)}</div>
                <div class="rec-meta">${h([r.categories, r.city].filter(Boolean).join(' · ').slice(0, 80))}</div>
                <div class="rec-reason">${h(r.reason || '')}</div>
            </div>
            <div class="rec-score">${r.stars ? r.stars + '★' : ''}<br><span style="color:var(--text-muted);font-size:11px">${r.score ? (r.score * 100).toFixed(0) + '%' : ''}</span></div>
        </div>
    `).join('') : '<div style="color:var(--text-muted);font-size:13px;padding:16px 0;text-align:center;">No recommendations returned.</div>';

    el.innerHTML = `
        ${data.search_intent ? `<div class="intent-chip">⟳ ${h(data.search_intent)}</div>` : ''}
        <div class="tag-row" style="margin-bottom:16px">${nb}${cb}</div>
        <div class="rec-list">${items}</div>
    `;
}

function h(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
