// js/config.js
// All UI-wide constants in one place.
//
// SINGLE ORIGIN: Task A and Task B are now served by one consolidated
// service. In local dev that's http://localhost:8080. In production it's
// your deployed domain. Leave API_BASE empty ('') to use a same-origin
// relative path — correct when the UI is served by the same host as the API.

// Local development default. For production, set this to your deployed
// API origin (e.g. 'https://bcthack.vancus.app') or leave '' if the UI
// is served from the same origin as the API.
export const API_BASE = 'http://localhost:8080';

// Endpoint paths on the consolidated service.
export const EP_HEALTH    = '/health';
export const EP_METADATA  = '/metadata';
export const EP_SIMULATE  = '/api/simulate';
export const EP_RECOMMEND = '/api/recommend';

// Client-side request timeout. Task B issues two LLM calls — generous headroom.
export const REQUEST_TIMEOUT_MS  = 180_000;   // 3 minutes
export const HEALTH_TIMEOUT_MS   = 4_000;
export const METADATA_TIMEOUT_MS = 6_000;

// Loading messages rotated during long requests
export const LOADING_MESSAGES_A = [
  'Building user persona…',
  'Predicting star rating…',
  'Selecting relevant past reviews…',
  'Generating review text…',
  'Applying cultural adapter…',
];

export const LOADING_MESSAGES_B = [
  'Reasoning about user intent…',
  'Embedding the search query…',
  'Retrieving business candidates…',
  'Reranking with persona context…',
  'Generating recommendations…',
];

export const LOADING_ROTATE_MS = 2400;
