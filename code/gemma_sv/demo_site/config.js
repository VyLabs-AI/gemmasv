// Safe repository default: the page uses its committed recorded replay.
// For local live testing, open:
//   http://127.0.0.1:8000/?api=http://127.0.0.1:8001
// A later deployment may replace this file with a public API URL. Never put
// credentials here: every visitor can read static JavaScript.
const query = new URLSearchParams(window.location.search);
const requestedApi = query.get("api") || "";
const localApi = /^http:\/\/(127\.0\.0\.1|localhost):\d+$/.test(requestedApi)
  ? requestedApi
  : "";
window.HERO_DEMO_CONFIG = Object.freeze({
  apiBase: localApi,
  forceReplay: !localApi,
  reviewMode: true,
});
