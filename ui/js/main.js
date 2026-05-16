// js/main.js  (v5 — polling health + city/state coupling)
// ---------------------------------------------------------------------------
// FIXES:
// 1. Health check polls every 5 seconds until both services are up,
//    re-fetching metadata when they come online. No more "you have to
//    refresh the page after the containers boot."
//
// 2. City/state coupled — selecting a city auto-fills its state, and
//    selecting a state filters available cities to that state.
//
// 3. Latency display in the toolbar shows last task's response time
//    so the demo audience can see the system is fast.

import { API_BASE } from './config.js';
import { $ } from './utils.js';
import * as dd from './dropdown.js';
import * as api from './api.js';
import { renderTaskA, renderTaskB, renderError } from './renderers.js';
import { renderTaskASkeleton, renderTaskBSkeleton, stopSkeletonAnimations } from './skeletons.js';

const state = {
  selectedStars: 4,
  conversationHistory: [],
  metadata: null,
  servicesOnline: { okA: false, okB: false },
  healthPollHandle: null,
};

const HEALTH_POLL_INTERVAL_MS = 5000;

// ── Boot ─────────────────────────────────────────────────────────────────────

async function boot() {
  setupStaticDropdowns();
  setupEventListeners();
  setupCursorGlow();
  setupTabIndicator();

  // Initial connection attempt
  await connectAndPopulate();

  // Keep polling if anything is offline
  if (!state.servicesOnline.okA || !state.servicesOnline.okB) {
    startHealthPolling();
  }
}

// ── Health polling ───────────────────────────────────────────────────────────

function startHealthPolling() {
  if (state.healthPollHandle) return;
  console.log('[health] Starting polling - one or more services offline');

  state.healthPollHandle = setInterval(async () => {
    const { okA, okB } = await api.healthCheck();
    const wasOffline = !state.servicesOnline.okA || !state.servicesOnline.okB;
    const nowOnline  = okA && okB;

    state.servicesOnline = { okA, okB };
    updateStatusDots(okA, okB);

    // First time both come online: fetch metadata + repopulate dropdowns
    if (wasOffline && nowOnline) {
      console.log('[health] Both services online! Fetching metadata...');
      const meta = await api.fetchMetadata({ okA, okB });
      if (meta) {
        state.metadata = meta;
        setupUserDropdowns();
        setupBusinessDropdown();
        setupCityStateDropdowns();
        updateStatusText(true, true);
      }
      stopHealthPolling();
    } else {
      updateStatusText(okA, okB);
    }
  }, HEALTH_POLL_INTERVAL_MS);
}

function stopHealthPolling() {
  if (state.healthPollHandle) {
    clearInterval(state.healthPollHandle);
    state.healthPollHandle = null;
    console.log('[health] Polling stopped - all systems go');
  }
}

// ── Static dropdowns ─────────────────────────────────────────────────────────

function setupStaticDropdowns() {
  dd.create('dd-b-topk', {
    options: [3, 5, 10].map(n => ({ value: String(n), label: `${n} recommendations` })),
    value: '5',
    placeholder: 'Top-k',
    searchable: false,
  });
  dd.create('dd-b-role', {
    options: [
      { value: 'user',      label: 'User'      },
      { value: 'assistant', label: 'Assistant' },
    ],
    value: 'user',
    placeholder: 'Role',
    searchable: false,
  });
}

// ── Data-dependent dropdowns ─────────────────────────────────────────────────

async function connectAndPopulate() {
  const { okA, okB } = await api.healthCheck();
  state.servicesOnline = { okA, okB };
  updateStatusDots(okA, okB);

  state.metadata = await api.fetchMetadata({ okA, okB });
  if (!state.metadata) {
    state.metadata = { top_users: [], top_cities: [], top_states: [], sample_businesses: [] };
  }

  setupUserDropdowns();
  setupBusinessDropdown();
  setupCityStateDropdowns();
  updateStatusText(okA, okB);
}

function updateStatusDots(okA, okB) {
  $('status-a').className = 'status-dot ' + (okA ? 'online' : 'offline');
  $('status-b').className = 'status-dot ' + (okB ? 'online' : 'offline');
}

function updateStatusText(okA, okB) {
  const txt = $('status-text');
  if (okA && okB) {
    txt.textContent = `Connected · ${state.metadata.top_users.length} users · ${state.metadata.sample_businesses.length} businesses`;
  } else if (state.healthPollHandle) {
    txt.textContent = 'Waiting for services to start...';
  } else {
    txt.textContent = 'One or more services offline';
  }
}

function setupUserDropdowns() {
  const opts = [
    { value: '__cold__', label: 'Cold-start user (no history)', sub: 'New user — uses dataset defaults' },
    ...state.metadata.top_users.map(u => ({
      value: u.user_id,
      label: `${u.user_id.slice(0, 12)}… · ${u.review_count} reviews`,
      sub: `${u.mean_stars?.toFixed?.(1) || '?'}★ avg · ${u.rating_bias_label || 'neutral'} rater`,
    })),
  ];
  ['dd-a-user', 'dd-b-user'].forEach(id => {
    dd.create(id, {
      options: opts,
      placeholder: 'Select a user…',
      value: '__cold__',
      onChange: (v) => {
        const hintEl = $(id === 'dd-a-user' ? 'a-uid-hint' : 'b-uid-hint');
        if (hintEl) hintEl.textContent = v === '__cold__' ? 'cold-start' : `${v.slice(0, 8)}…`;
      },
    });
  });
}

function setupBusinessDropdown() {
  const opts = [
    { value: '__custom__', label: 'Custom (fill fields manually)', sub: 'Use the inputs below' },
    ...state.metadata.sample_businesses.map(b => ({
      value: b.business_id,
      label: b.biz_name,
      sub: `${(b.categories || '').slice(0, 50)} · ${b.city}, ${b.state}`,
    })),
  ];
  const initVal = state.metadata.sample_businesses[0]?.business_id || '__custom__';

  dd.create('dd-a-biz', {
    options: opts,
    placeholder: 'Pick a sample business…',
    value: initVal,
    onChange: (v) => {
      if (v === '__custom__') return;
      const biz = state.metadata.sample_businesses.find(b => b.business_id === v);
      if (!biz) return;
      $('a-name').value = biz.biz_name || '';
      $('a-cats').value = biz.categories || '';
      // When user picks a sample business, sync city/state from it
      dd.setValue('dd-a-city', biz.city || '__custom__');
      dd.setValue('dd-a-state', biz.state || '__custom__');
      setStars(Math.round(biz.biz_avg_stars || 4));
    },
  });

  if (initVal !== '__custom__') {
    const biz = state.metadata.sample_businesses.find(b => b.business_id === initVal);
    if (biz) {
      $('a-name').value = biz.biz_name || '';
      $('a-cats').value = biz.categories || '';
    }
  }
}

// ── City / State dropdowns with coupling ─────────────────────────────────────
//
// When a city is picked, auto-fill its state.
// When a state is picked, filter cities to only show those in that state
// (plus keep the synthetic Nigerian demos available always).

function setupCityStateDropdowns() {
  const cities = state.metadata.top_cities || [];
  const states = state.metadata.top_states || [];

  // Build city -> state map from top_cities metadata
  const cityToState = new Map();
  cities.forEach(c => cityToState.set(c.city, c.state));
  // Manual Nigerian entries
  cityToState.set('Lagos', 'LA');
  cityToState.set('Abuja', 'FC');

  // Build state -> [cities] reverse map
  const stateToCities = new Map();
  cityToState.forEach((st, ci) => {
    if (!stateToCities.has(st)) stateToCities.set(st, []);
    stateToCities.get(st).push(ci);
  });

  const cityOpts = [
    { value: '__custom__', label: 'Custom', sub: '' },
    ...cities.map(c => ({
      value: c.city, label: c.city,
      sub: `${c.state} · ${c.count?.toLocaleString?.() || ''} reviews`,
    })),
    { value: 'Lagos', label: 'Lagos', sub: 'LA · Nigerian demo' },
    { value: 'Abuja', label: 'Abuja', sub: 'FC · Nigerian demo' },
  ];

  const stateOpts = [
    { value: '__custom__', label: 'Custom', sub: '' },
    ...states.map(s => ({
      value: s.state, label: s.state,
      sub: `${s.count?.toLocaleString?.() || ''} reviews`,
    })),
    { value: 'LA', label: 'LA', sub: 'Lagos demo' },
    { value: 'FC', label: 'FC', sub: 'Abuja FCT demo' },
  ];

  dd.create('dd-a-city', {
    options: cityOpts,
    placeholder: 'Pick a city…',
    value: 'Lagos',
    onChange: (city) => {
      // When city changes, sync state to its known state (if any)
      if (city && city !== '__custom__' && cityToState.has(city)) {
        const matchedState = cityToState.get(city);
        if (matchedState && dd.getValue('dd-a-state') !== matchedState) {
          dd.setValue('dd-a-state', matchedState);
        }
      }
    },
  });

  dd.create('dd-a-state', {
    options: stateOpts,
    placeholder: 'Pick a state…',
    value: 'LA',
    onChange: (st) => {
      // When state changes, filter city options to only those in this state
      if (!st || st === '__custom__') {
        // Reset to all cities
        dd.setOptions('dd-a-city', cityOpts);
        return;
      }
      const validCities = stateToCities.get(st) || [];
      const filtered = [
        { value: '__custom__', label: 'Custom', sub: '' },
        ...validCities.map(ci => {
          const cityMeta = cities.find(c => c.city === ci);
          if (cityMeta) {
            return {
              value: ci, label: ci,
              sub: `${st} · ${cityMeta.count?.toLocaleString?.() || ''} reviews`,
            };
          }
          return { value: ci, label: ci, sub: `${st} · demo` };
        }),
      ];
      dd.setOptions('dd-a-city', filtered);

      // If current city is not in this state, reset to first valid city
      const currentCity = dd.getValue('dd-a-city');
      if (currentCity !== '__custom__' && !validCities.includes(currentCity)) {
        if (validCities.length > 0) {
          dd.setValue('dd-a-city', validCities[0]);
        } else {
          dd.setValue('dd-a-city', '__custom__');
        }
      }
    },
  });
}

// ── Event wiring ─────────────────────────────────────────────────────────────

function setupEventListeners() {
  document.querySelectorAll('.tab').forEach(t => {
    t.addEventListener('click', () => switchTab(t.dataset.tab));
  });
  document.querySelectorAll('.star-btn').forEach(b => {
    b.addEventListener('click', () => setStars(parseInt(b.dataset.star, 10)));
  });
  $('a-btn').addEventListener('click', runSimulate);
  $('b-btn').addEventListener('click', runRecommend);
  $('b-add-turn').addEventListener('click', addTurn);
  $('b-clear-turns').addEventListener('click', clearTurns);
  $('b-turn-text').addEventListener('keydown', e => {
    if (e.key === 'Enter') addTurn();
  });
}

function setupTabIndicator() {
  requestAnimationFrame(positionTabIndicator);
  window.addEventListener('resize', positionTabIndicator);
}

function positionTabIndicator() {
  const indicator = $('tab-indicator');
  const active = document.querySelector('.tab.active');
  if (!indicator || !active) return;
  const parent = active.parentElement.getBoundingClientRect();
  const rect = active.getBoundingClientRect();
  indicator.style.left = `${rect.left - parent.left}px`;
  indicator.style.width = `${rect.width}px`;
}

function switchTab(t) {
  document.querySelectorAll('.tab').forEach(el => {
    el.classList.toggle('active', el.dataset.tab === t);
  });
  $('panel-a').classList.toggle('active', t === 'a');
  $('panel-b').classList.toggle('active', t === 'b');
  positionTabIndicator();
}

function setupCursorGlow() {
  const glow = $('cursor-glow');
  if (!glow) return;

  let rafId = null;
  let targetX = 0, targetY = 0;
  let currentX = 0, currentY = 0;

  document.addEventListener('mousemove', e => {
    targetX = e.clientX;
    targetY = e.clientY;
    glow.style.opacity = '1';
    if (!rafId) animate();
  });

  document.addEventListener('mouseleave', () => {
    glow.style.opacity = '0';
  });

  function animate() {
    currentX += (targetX - currentX) * 0.08;
    currentY += (targetY - currentY) * 0.08;
    glow.style.left = `${currentX}px`;
    glow.style.top  = `${currentY}px`;
    if (Math.abs(targetX - currentX) > 0.5 || Math.abs(targetY - currentY) > 0.5) {
      rafId = requestAnimationFrame(animate);
    } else {
      rafId = null;
    }
  }
}

function setStars(n) {
  state.selectedStars = n;
  document.querySelectorAll('.star-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.star, 10) === n);
  });
}

function addTurn() {
  const role = dd.getValue('dd-b-role') || 'user';
  const text = $('b-turn-text').value.trim();
  if (!text) return;
  state.conversationHistory.push({ role, content: text });
  $('b-turn-text').value = '';
  renderConvo();
}

function clearTurns() {
  state.conversationHistory = [];
  renderConvo();
}

function renderConvo() {
  const el = $('b-convo-display');
  if (!state.conversationHistory.length) {
    el.innerHTML = '<div class="convo-empty">No turns yet</div>';
    return;
  }
  el.innerHTML = state.conversationHistory.map(t => `
    <div class="convo-turn">
      <span class="convo-role ${t.role}">${t.role}</span>
      <span>${escHtmlInline(t.content)}</span>
    </div>`).join('');
  el.scrollTop = el.scrollHeight;
}

function escHtmlInline(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Task A request ───────────────────────────────────────────────────────────

async function runSimulate() {
  const btn = $('a-btn'), result = $('a-result');
  setButtonLoading(btn, true, 'SIMULATING…');
  renderTaskASkeleton(result);

  const userVal  = dd.getValue('dd-a-user')  || '__cold__';
  const bizVal   = dd.getValue('dd-a-biz')   || '__custom__';
  const cityVal  = dd.getValue('dd-a-city');
  const stateVal = dd.getValue('dd-a-state');

  const payload = {
    user_id:     userVal === '__cold__' ? `cold_user_${Date.now()}` : userVal,
    business_id: bizVal  === '__custom__' ? `custom_biz_${Date.now()}` : bizVal,
    product_details: {
      name:         $('a-name').value.trim(),
      categories:   $('a-cats').value.trim(),
      city:         cityVal  === '__custom__' ? '' : (cityVal  || ''),
      state:        stateVal === '__custom__' ? '' : (stateVal || ''),
      stars:        state.selectedStars,
      review_count: 88,
    },
  };

  const t0 = performance.now();
  try {
    const data = await api.simulateReview(payload);
    const latency = performance.now() - t0;
    stopSkeletonAnimations();
    renderTaskA(data, result, latency);
    updateLatencyMeta('A', latency);
  } catch (e) {
    stopSkeletonAnimations();
    renderError(e, result, API_BASE);
  } finally {
    setButtonLoading(btn, false, '▶ SIMULATE REVIEW');
  }
}

// ── Task B request ───────────────────────────────────────────────────────────

async function runRecommend() {
  const btn = $('b-btn'), result = $('b-result');
  const topK = parseInt(dd.getValue('dd-b-topk') || '5', 10);

  setButtonLoading(btn, true, 'REASONING…');
  renderTaskBSkeleton(result, topK);

  const userVal = dd.getValue('dd-b-user') || '__cold__';
  const payload = {
    user_id: userVal === '__cold__' ? `cold_user_${Date.now()}` : userVal,
    context: $('b-ctx').value.trim(),
    conversation_history: state.conversationHistory,
    top_k: topK,
  };

  const t0 = performance.now();
  try {
    const data = await api.recommend(payload);
    const latency = performance.now() - t0;
    stopSkeletonAnimations();
    renderTaskB(data, result, latency);
    updateLatencyMeta('B', latency);
  } catch (e) {
    stopSkeletonAnimations();
    renderError(e, result, API_BASE);
  } finally {
    setButtonLoading(btn, false, '▶ GET RECOMMENDATIONS');
  }
}

function setButtonLoading(btn, loading, defaultLabel) {
  const labelEl = btn.querySelector('.btn-label');
  btn.disabled = loading;
  if (loading) {
    labelEl.innerHTML = `<span class="spinner-inline"></span>${defaultLabel}`;
  } else {
    labelEl.textContent = defaultLabel;
  }
}

function updateLatencyMeta(task, latencyMs) {
  const meta = $('latency-meta');
  if (!meta) return;
  const seconds = (latencyMs / 1000).toFixed(1);
  meta.textContent = `Task ${task} · ${seconds}s`;
}

// ── Go ───────────────────────────────────────────────────────────────────────
boot();
