// js/api.js
// Typed API wrappers with timeouts and proper error propagation.

import { API_A, API_B, HEALTH_TIMEOUT_MS, METADATA_TIMEOUT_MS } from './config.js';
import { fetchWithTimeout, postJson } from './utils.js';

export async function healthCheck() {
  let okA = false, okB = false;
  try {
    const r = await fetchWithTimeout(`${API_A}/health`, {}, HEALTH_TIMEOUT_MS);
    okA = r.ok;
  } catch (_) { /* offline */ }
  try {
    const r = await fetchWithTimeout(`${API_B}/health`, {}, HEALTH_TIMEOUT_MS);
    okB = r.ok;
  } catch (_) { /* offline */ }
  return { okA, okB };
}

export async function fetchMetadata({ okA, okB }) {
  if (!okA && !okB) return null;
  try {
    const url = `${okA ? API_A : API_B}/metadata`;
    const r = await fetchWithTimeout(url, {}, METADATA_TIMEOUT_MS);
    if (!r.ok) return null;
    return await r.json();
  } catch (_) {
    return null;
  }
}

export async function simulateReview(payload) {
  return postJson(`${API_A}/simulate-review`, payload);
}

export async function recommend(payload) {
  return postJson(`${API_B}/recommend`, payload);
}
