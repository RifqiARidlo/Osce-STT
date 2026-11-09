// Utilitas bersama: fetch API (+token), helper DOM, header navigasi.
const TOKEN_KEY = "osce_access_token";

export async function api(path, { method = "GET", json, form } = {}) {
  const headers = {};
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers["X-Access-Token"] = token;
  let body;
  if (json !== undefined) { headers["Content-Type"] = "application/json"; body = JSON.stringify(json); }
  if (form !== undefined) body = form;
  const res = await fetch(path, { method, headers, body });
  if (res.status === 401) {
    const t = prompt("Server meminta token akses (APP_ACCESS_TOKEN):");
    if (t) { localStorage.setItem(TOKEN_KEY, t.trim()); return api(path, { method, json, form }); }
  }
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
    throw new Error(`${res.status}: ${detail}`);
  }
  return data;
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "value") node.value = v; // textarea/select tidak membaca atribut value
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
    else if (k in node && typeof v !== "string") node[k] = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

export const pct = (x) => `${(100 * (x || 0)).toFixed(0)}%`;
export const fmtTime = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function notice(container, msg, kind = "") {
  container.replaceChildren(el("div", { class: `notice ${kind}`, text: msg }));
}

export function header(active) {
  const links = [["/", "Rekam Sesi"], ["/enroll", "Enrollment Dokter"], ["/cases", "Bank Kasus OSCE"], ["/metrics", "Evaluasi ML"]];
  const health = el("span", { class: "health", text: "memeriksa server…" });
  const bar = el("header", { class: "top" },
    el("span", { class: "title", text: "Medical STT · OSCE Case Bank" }),
    el("nav", {}, links.map(([href, label]) => el("a", { href, class: href === active ? "active" : "", text: label }))),
    health);
  document.body.prepend(bar);
  api("/api/health").then((h) => {
    health.textContent = `Whisper ${h.whisper_model} · LLM ${h.llm_backend} · speaker: ${h.speaker_model} · intent: ${h.intent_model}`;
  }).catch((e) => { health.textContent = `server tidak terjangkau (${e.message})`; });
}

const MY_JOBS = "osce_my_jobs";
export function rememberJob(id) {
  const jobs = JSON.parse(localStorage.getItem(MY_JOBS) || "[]").filter((j) => j !== id);
  jobs.unshift(id);
  localStorage.setItem(MY_JOBS, JSON.stringify(jobs.slice(0, 20)));
}
export const myJobs = () => JSON.parse(localStorage.getItem(MY_JOBS) || "[]");
