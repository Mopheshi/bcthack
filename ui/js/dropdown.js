// js/dropdown.js  (v4 — portaled menu)
// ---------------------------------------------------------------------------
// FIX for z-index/stacking issues:
//   The card containers use backdrop-filter, which creates new stacking
//   contexts that trap dropdown menus. v4 portals the menu element to
//   document.body using fixed positioning, escaping the stacking trap.
//
// The trigger button remains inside the card; only the menu floats.

import { escHtml, escAttr, $ } from './utils.js';

const state = {};   // elId -> {value, options, placeholder, onChange, searchable, search, opened, menuEl}

export function create(elId, opts) {
  const el = $(elId);
  if (!el) return;

  state[elId] = {
    value:       opts.value ?? null,
    options:     opts.options ?? [],
    placeholder: opts.placeholder ?? 'Select…',
    onChange:    opts.onChange,
    searchable:  opts.searchable !== false,
    search:      '',
    opened:      false,
    menuEl:      null,   // created on first open
  };
  renderTrigger(elId);
}

export function setOptions(elId, options) {
  if (!state[elId]) return;
  state[elId].options = options;
  if (state[elId].opened) renderMenu(elId);
}

export function setValue(elId, value) {
  if (!state[elId]) return;
  state[elId].value = value;
  renderTrigger(elId);
}

export function getValue(elId) {
  return state[elId]?.value ?? null;
}

// ── Trigger button (inside the card) ─────────────────────────────────────────
function renderTrigger(elId) {
  const el = $(elId);
  const s  = state[elId];
  if (!el || !s) return;

  const selected = s.options.find(o => o.value === s.value);
  const triggerLabel = selected ? selected.label : s.placeholder;
  const triggerCls   = selected ? 'dd-trigger' : 'dd-trigger placeholder';

  el.className = 'dd' + (s.opened ? ' open' : '');
  el.innerHTML = `
    <button type="button" class="${triggerCls}" data-dd-toggle="${elId}">${escHtml(triggerLabel)}</button>
    <span class="dd-arrow">▼</span>`;

  el.querySelector('[data-dd-toggle]')?.addEventListener('click', e => {
    e.stopPropagation();
    toggle(elId);
  });
}

// ── Menu (portaled to document.body) ─────────────────────────────────────────
function renderMenu(elId) {
  const el = $(elId);
  const s  = state[elId];
  if (!el || !s) return;

  if (!s.menuEl) {
    s.menuEl = document.createElement('div');
    s.menuEl.className = 'dd-menu-portal';
    document.body.appendChild(s.menuEl);
  }

  const filtered = s.search
    ? s.options.filter(o => (o.label || '').toLowerCase().includes(s.search.toLowerCase()))
    : s.options;

  const searchHtml = s.searchable && s.options.length > 6
    ? `<input class="dd-search" placeholder="Search…" data-dd-search="${elId}" value="${escAttr(s.search)}" />`
    : '';

  const optionsHtml = filtered.length === 0
    ? `<div class="dd-empty">No matches</div>`
    : filtered.map(o => `
        <div class="dd-option${o.value === s.value ? ' selected' : ''}"
             data-dd-select="${elId}"
             data-value="${escAttr(o.value)}">
          <span>${escHtml(o.label)}</span>
          ${o.sub ? `<span class="dd-option-sub">${escHtml(o.sub)}</span>` : ''}
        </div>`).join('');

  s.menuEl.innerHTML = searchHtml + optionsHtml;

  // Position the menu under the trigger using viewport coordinates
  positionMenu(elId);

  // Wire up search and selection
  const searchInput = s.menuEl.querySelector('[data-dd-search]');
  if (searchInput) {
    searchInput.addEventListener('input', e => {
      s.search = e.target.value;
      renderMenu(elId);
    });
    searchInput.addEventListener('click', e => e.stopPropagation());
    requestAnimationFrame(() => {
      searchInput.focus();
      const len = searchInput.value.length;
      searchInput.setSelectionRange(len, len);
    });
  }

  s.menuEl.querySelectorAll('[data-dd-select]').forEach(opt => {
    opt.addEventListener('click', e => {
      e.stopPropagation();
      select(elId, opt.dataset.value);
    });
  });

  // Visibility
  s.menuEl.classList.toggle('open', s.opened);
}

function positionMenu(elId) {
  const el = $(elId);
  const s  = state[elId];
  if (!el || !s?.menuEl) return;

  const rect = el.getBoundingClientRect();
  const menuWidth = rect.width;

  s.menuEl.style.position = 'fixed';
  s.menuEl.style.top  = `${rect.bottom + 6}px`;
  s.menuEl.style.left = `${rect.left}px`;
  s.menuEl.style.width = `${menuWidth}px`;

  // If menu would overflow viewport bottom, flip above
  const menuRect = s.menuEl.getBoundingClientRect();
  if (rect.bottom + menuRect.height + 12 > window.innerHeight) {
    s.menuEl.style.top = `${rect.top - menuRect.height - 6}px`;
  }
}

function toggle(elId) {
  // Close all others
  for (const id in state) {
    if (id !== elId && state[id].opened) {
      state[id].opened = false;
      if (state[id].menuEl) state[id].menuEl.classList.remove('open');
      renderTrigger(id);
    }
  }
  state[elId].opened = !state[elId].opened;
  if (state[elId].opened) state[elId].search = '';
  renderTrigger(elId);
  if (state[elId].opened) renderMenu(elId);
  else if (state[elId].menuEl) state[elId].menuEl.classList.remove('open');
}

function select(elId, value) {
  const s = state[elId];
  s.value  = value;
  s.opened = false;
  renderTrigger(elId);
  if (s.menuEl) s.menuEl.classList.remove('open');
  if (s.onChange) s.onChange(value, s.options.find(o => o.value === value));
}

// ── Global handlers ──────────────────────────────────────────────────────────

// Close on outside click
document.addEventListener('click', () => {
  let dirty = false;
  for (const id in state) {
    if (state[id].opened) {
      state[id].opened = false;
      if (state[id].menuEl) state[id].menuEl.classList.remove('open');
      dirty = true;
    }
  }
  if (dirty) for (const id in state) renderTrigger(id);
});

// Reposition open menus on scroll/resize
window.addEventListener('scroll', () => {
  for (const id in state) {
    if (state[id].opened) positionMenu(id);
  }
}, { passive: true });

window.addEventListener('resize', () => {
  for (const id in state) {
    if (state[id].opened) positionMenu(id);
  }
});
