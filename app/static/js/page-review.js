import { api, el, header, notice, pct } from "./api.js";
import { renderSchemaForm } from "./schema-form.js";

header("");
const $ = (id) => document.getElementById(id);
const jobId = new URLSearchParams(location.search).get("job");
const INTENTS = ["S_KELUHAN", "O_PEMERIKSAAN", "A_DIAGNOSIS", "P_EDUKASI_TERAPI", "CHITCHAT"];
const STAGES = [["stage1_preprocess", "1 · Denoise + VAD"], ["stage2_asr", "2a · ASR Whisper"],
  ["stage2_speaker", "2b · Speaker Role ML"], ["stage3_intent", "3 · Intent ML"],
  ["stage4_summarize", "4 · LLM terstandar"], ["stage5_finalize", "5 · Finalisasi"]];
const LOW_CONF = 0.65;

let utterances = [];
let form = null;
let readOnly = false;

$("toggle-ml").onchange = (e) => document.body.classList.toggle("hide-ml", !e.target.checked);

function stepper(stage) {
  const idx = STAGES.findIndex(([k]) => k === stage);
  return el("div", { class: "stepper" }, STAGES.map(([k, label], i) =>
    el("span", { class: i < idx ? "done" : i === idx ? "active" : "", text: label })));
}

function flash(id) {
  const node = document.querySelector(`[data-utt="${id}"]`);
  if (!node) return;
  node.scrollIntoView({ behavior: "smooth", block: "center" });
  node.classList.add("flash");
  setTimeout(() => node.classList.remove("flash"), 1500);
}

function confBadge(value, source) {
  const cls = value < LOW_CONF ? "conf low" : "conf";
  return el("span", { class: `${cls} ml-only`, title: `sumber: ${source}`, text: `${pct(value)} ${source === "model" ? "" : `(${source})`}`.trim() });
}

function bubble(u) {
  const roleBtn = el("button", { class: "small", text: u.role, disabled: readOnly, title: "Tukar peran",
    onclick: () => { u.role = u.role === "DOKTER" ? "PASIEN" : "DOKTER"; u.role_source = "human"; redrawChat(); } });
  const intentSel = el("select", { disabled: readOnly, onchange: (e) => { u.intent = e.target.value; u.intent_source = "human"; redrawChat(); } },
    INTENTS.map((i) => el("option", { value: i, text: i, selected: u.intent === i })));
  const probs = el("div", { class: "probs ml-only" }, Object.entries(u.intent_probs || {})
    .sort((a, b) => b[1] - a[1]).flatMap(([k, v]) => [
      el("span", { text: k }), el("div", { class: "bar", style: `width:${(v * 100).toFixed(0)}%` }), el("span", { class: "mono", text: pct(v) })]));
  const lex = (u.lexicon_hits || []).length
    ? el("div", { class: "muted ml-only", text: `lexicon: ${u.lexicon_hits.join("; ")}` }) : null;
  return el("div", { class: `bubble ${u.role} ${u.intent === "CHITCHAT" ? "CHITCHAT-dim" : ""}`, "data-utt": u.id },
    el("div", { class: "meta" },
      el("span", { class: "mono", text: `#${u.id}` }), roleBtn, confBadge(u.role_confidence, u.role_source),
      intentSel, confBadge(u.intent_confidence, u.intent_source),
      el("span", { class: "muted mono", text: `${u.start.toFixed(1)}–${u.end.toFixed(1)}s` }),
      el("span", { class: "muted mono ml-only", text: `ASR ${pct(u.asr_confidence)}` })),
    el("textarea", { rows: Math.max(1, Math.ceil(u.text.length / 70)), value: u.text, readOnly,
      oninput: (e) => { u.text = e.target.value; } }),
    lex, probs);
}

function redrawChat() {
  $("chat").replaceChildren(...utterances.map(bubble));
  const n = utterances.length || 1;
  const lowRole = utterances.filter((u) => u.role_source !== "human" && u.role_confidence < LOW_CONF).length;
  const lowIntent = utterances.filter((u) => u.intent_source !== "human" && u.intent_confidence < LOW_CONF).length;
  const counts = INTENTS.map((i) => `${i}: ${utterances.filter((u) => u.intent === i).length}`).join(" · ");
  $("ml-summary").textContent = `${utterances.length} utterance · confidence rendah — peran: ${lowRole} (${pct(lowRole / n)}), ` +
    `kelas: ${lowIntent} (${pct(lowIntent / n)}) · ${counts}`;
}

function payload() { return { utterances, summary: form.getValue() }; }

async function load() {
  if (!jobId) { notice($("status"), "Parameter ?job= tidak ada.", "err"); return; }
  const job = await api(`/api/jobs/${jobId}`);
  $("title").textContent = `Review sesi ${job.id}`;
  if (["queued", "processing"].includes(job.status)) {
    $("status").replaceChildren(el("div", { class: "notice", text:
      job.status === "queued" ? `Menunggu di antrean (posisi ${job.position}).` : "Sedang diproses…" }), stepper(job.stage));
    setTimeout(load, 2000);
    return;
  }
  if (job.status === "failed") { notice($("status"), `Job gagal: ${job.error}`, "err"); return; }

  const templates = await api("/api/templates");
  const tpl = templates.find((t) => t.id === job.template_id);
  const source = job.review || job.result;
  utterances = structuredClone(source.utterances);
  readOnly = job.status === "verified";
  $("form-title").textContent = `Rangkuman terstandar — ${tpl.title}`;
  $("status").replaceChildren();
  if (readOnly) notice($("status"), `Sudah diverifikasi & diekspor sebagai kasus OSCE ${job.case_id}.`, "ok");
  else if (job.review) notice($("status"), "Menampilkan draft review tersimpan.");

  const warnings = [...(job.result.warnings || [])];
  if (job.result.summary_source === "fallback_rule") warnings.unshift("Rangkuman berasal dari fallback rule-based, bukan LLM.");
  if ((job.result.pii_hits || []).length) warnings.push(`PII dianonimkan otomatis: ${job.result.pii_hits.join(", ")}. Periksa sisa nama/alamat.`);
  $("warnings").replaceChildren(...warnings.map((w) => el("div", { class: "notice", text: w })));

  form = renderSchemaForm(tpl.schema, source.summary, { onSourceClick: flash, readOnly });
  $("form").replaceChildren(form.element);
  $("timings").textContent = JSON.stringify({ duration_s: job.result.duration_s, timings: job.result.timings, models: job.result.models }, null, 2);
  $("btn-draft").disabled = readOnly; $("btn-verify").disabled = readOnly;
  $("content").style.display = "";
  redrawChat();
}

$("btn-draft").onclick = async () => {
  try { await api(`/api/jobs/${jobId}/review`, { method: "PUT", json: payload() }); notice($("save-msg"), "Draft tersimpan.", "ok"); }
  catch (e) { notice($("save-msg"), e.message, "err"); }
};

$("btn-verify").onclick = async () => {
  if (!confirm("Kunci data ini ke Bank Kasus OSCE? Koreksi label juga masuk ke training_data (active learning).")) return;
  try {
    const res = await api(`/api/jobs/${jobId}/verify`, { method: "POST", json: payload() });
    notice($("save-msg"), `Terverifikasi. Kasus OSCE: ${res.case_id}`, "ok");
    load();
  } catch (e) { notice($("save-msg"), e.message, "err"); }
};

load().catch((e) => notice($("status"), e.message, "err"));
