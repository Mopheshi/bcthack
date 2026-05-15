// js/utils.js
// Pure helpers - no state, no DOM coupling beyond the helpers themselves.

import { REQUEST_TIMEOUT_MS } from './config.js';

export const $  = (id) => document.getElementById(id);
export const qs = (sel, parent = document) => parent.querySelector(sel);

export function escHtml(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function escAttr(s) {
  return String(s ?? '')
    .replace(/'/g, "\\'")
    .replace(/"/g, '&quot;');
}

/**
 * Fetch with built-in AbortController timeout and clean error message.
 */
export async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const res = await fetch(url, { ...options, signal: ctrl.signal });
    clearTimeout(timer);
    return res;
  } catch (e) {
    clearTimeout(timer);
    if (e.name === 'AbortError') {
      const err = new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds`);
      err.timeout = true;
      throw err;
    }
    throw e;
  }
}

/**
 * POST JSON with timeout and structured error.
 */
export async function postJson(url, body, timeoutMs = REQUEST_TIMEOUT_MS) {
  const res = await fetchWithTimeout(
    url,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    },
    timeoutMs
  );

  if (!res.ok) {
    const errText = await res.text().catch(() => '');
    const err = new Error(`HTTP ${res.status}: ${errText.slice(0, 300)}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

/**
 * Format milliseconds as "1.2s" or "850ms".
 */
export function formatLatency(ms) {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
