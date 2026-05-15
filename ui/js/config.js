// js/config.js
// All UI-wide constants in one place.

export const API_A = 'http://localhost:8001';
export const API_B = 'http://localhost:8002';

// Client-side request timeout. Task B can do 3 LLM calls so we allow generous headroom.
export const REQUEST_TIMEOUT_MS = 180_000;   // 3 minutes
export const HEALTH_TIMEOUT_MS  = 3_000;
export const METADATA_TIMEOUT_MS = 5_000;

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
  'Retrieving 20 business candidates…',
  'Reranking with persona context…',
  'Generating recommendations…',
];

// Rotate every N ms during a request
export const LOADING_ROTATE_MS = 2400;
