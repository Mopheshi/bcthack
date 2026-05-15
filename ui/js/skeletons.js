// js/skeletons.js
// Skeleton shimmer templates for loading states.

import { LOADING_MESSAGES_A, LOADING_MESSAGES_B, LOADING_ROTATE_MS } from './config.js';
import { escHtml } from './utils.js';

let rotateTimer = null;
let elapsedTimer = null;
let startTime = 0;

/**
 * Task A skeleton: mimics the final review-with-tags layout.
 */
export function renderTaskASkeleton(container) {
  container.innerHTML = `
    <div class="loading-strip">
      <div class="loading-strip-spinner"></div>
      <div class="loading-strip-text" id="ldg-msg-a">${escHtml(LOADING_MESSAGES_A[0])}</div>
      <div class="loading-strip-time" id="ldg-time-a">0.0s</div>
    </div>
    <div class="skeleton-task-a">
      <div class="sk-stars-row">
        <div class="skeleton sk-star-val"></div>
        <div class="skeleton sk-stars-glyphs"></div>
        <div class="sk-confidence">
          <div class="skeleton sk-confidence-label"></div>
          <div class="skeleton sk-confidence-bar"></div>
        </div>
      </div>
      <div class="sk-review-text">
        <div class="skeleton sk-line sk-w-100"></div>
        <div class="skeleton sk-line sk-w-90"></div>
        <div class="skeleton sk-line sk-w-100"></div>
        <div class="skeleton sk-line sk-w-80"></div>
        <div class="skeleton sk-line sk-w-70"></div>
      </div>
      <div class="sk-tag-row">
        <div class="skeleton sk-tag"></div>
        <div class="skeleton sk-tag"></div>
        <div class="skeleton sk-tag"></div>
        <div class="skeleton sk-tag"></div>
      </div>
    </div>`;

  startMessageRotation('ldg-msg-a', 'ldg-time-a', LOADING_MESSAGES_A);
}

/**
 * Task B skeleton: mimics the recommendation list layout.
 */
export function renderTaskBSkeleton(container, topK = 5) {
  const recItems = Array.from({ length: topK }, () => `
    <div class="sk-rec-item">
      <div class="skeleton sk-rec-rank"></div>
      <div class="sk-rec-body">
        <div class="skeleton sk-rec-name"></div>
        <div class="skeleton sk-rec-meta"></div>
        <div class="skeleton sk-rec-reason"></div>
        <div class="skeleton sk-rec-reason-2"></div>
      </div>
      <div class="sk-rec-score">
        <div class="skeleton sk-rec-score-val"></div>
        <div class="skeleton sk-rec-score-pct"></div>
      </div>
    </div>
  `).join('');

  container.innerHTML = `
    <div class="loading-strip">
      <div class="loading-strip-spinner"></div>
      <div class="loading-strip-text" id="ldg-msg-b">${escHtml(LOADING_MESSAGES_B[0])}</div>
      <div class="loading-strip-time" id="ldg-time-b">0.0s</div>
    </div>
    <div class="skeleton-task-b">
      <div class="skeleton sk-intent-chip"></div>
      <div class="sk-status-tags">
        <div class="skeleton sk-tag"></div>
        <div class="skeleton sk-tag"></div>
      </div>
      <div class="sk-rec-list">${recItems}</div>
    </div>`;

  startMessageRotation('ldg-msg-b', 'ldg-time-b', LOADING_MESSAGES_B);
}

function startMessageRotation(msgElId, timeElId, messages) {
  stopRotation();
  startTime = performance.now();
  let i = 0;

  rotateTimer = setInterval(() => {
    i = (i + 1) % messages.length;
    const el = document.getElementById(msgElId);
    if (el) el.textContent = messages[i];
  }, LOADING_ROTATE_MS);

  elapsedTimer = setInterval(() => {
    const el = document.getElementById(timeElId);
    if (el) {
      const elapsed = (performance.now() - startTime) / 1000;
      el.textContent = `${elapsed.toFixed(1)}s`;
    }
  }, 100);
}

export function stopSkeletonAnimations() {
  stopRotation();
}

function stopRotation() {
  if (rotateTimer) { clearInterval(rotateTimer); rotateTimer = null; }
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
}
