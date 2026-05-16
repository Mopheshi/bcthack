// js/api.js
// Typed API wrappers — single consolidated origin.

import {
  API_BASE, EP_HEALTH, EP_METADATA, EP_SIMULATE, EP_RECOMMEND,
  HEALTH_TIMEOUT_MS, METADATA_TIMEOUT_MS,
} from './config.js';
import { fetchWithTimeout, postJson } from './utils.js';

const url = (path) => `${API_BASE}${path}`;

/**
 * Health check. The consolidated service is a single origin, so there is
 * one health state. We still return {okA, okB} so existing UI code that
 * tracks two dots keeps working — both reflect the same service.
 */
export async function healthCheck() {
  let ok = false;
  try {
    const r = await fetchWithTimeout(url(EP_HEALTH), {}, HEALTH_TIMEOUT_MS);
    if (r.ok) {
      const body = await r.json().catch(() => ({}));
      ok = body.ready === true || body.status === 'ok';
    }
  } catch (_) { /* offline */ }
  return { okA: ok, okB: ok, ok };
}

export async function fetchMetadata(health) {
  if (!health || !(health.ok ?? (health.okA && health.okB))) return null;
  try {
    const r = await fetchWithTimeout(url(EP_METADATA), {}, METADATA_TIMEOUT_MS);
    if (!r.ok) return null;
    return await r.json();
  } catch (_) {
    return null;
  }
}

export async function simulateReview(payload) {
  return postJson(url(EP_SIMULATE), payload);
}

export async function recommend(payload) {
  return postJson(url(EP_RECOMMEND), payload);
}
