import { api, el, fmtTime, header, myJobs, notice, rememberJob } from "./api.js";
import { WavRecorder } from "./recorder.js";

header("/");
const $ = (id) => document.getElementById(id);
let templates = [];
let recorder = null;
let tick = null;

async function loadConfig() {
  const [doctors, tpls] = await Promise.all([api("/api/doctors"), api("/api/templates")]);
  templates = tpls;
  $("doctor").replaceChildren(
    el("option", { value: "", text: "— tanpa profil (peran dikoreksi manual) —" }),
    ...doctors.map((d) => el("option", { value: d.id, text: `${d.name} (${d.enroll_seconds}s)` })));
  const saved = localStorage.getItem("osce_doctor");
  if (saved && doctors.some((d) => d.id === saved)) $("doctor").value = saved;
  $("template").replaceChildren(...tpls.map((t) => el("option", { value: t.id, text: t.title })));
  const savedTpl = localStorage.getItem("osce_template");
  if (savedTpl && tpls.some((t) => t.id === savedTpl)) $("template").value = savedTpl;
  renderLocked();
}

function renderLocked() {
  const tpl = templates.find((t) => t.id === $("template").value);
  if (!tpl) return;
  $("template-desc").textContent = tpl.description;
  const box = $("locked");
  box.replaceChildren(...tpl.lockable.map((key) => {
    const opts = tpl.schema.properties[key].enum;
    return el("div", {}, el("label", { text: key.replace(/_/g, " ") }),
      el("select", { "data-key": key }, el("option", { value: "", text: "(otomatis dari percakapan)" }),
        opts.filter((o) => o !== "Tidak disebutkan").map((o) => el("option", { value: o, text: o }))));
  }));
}

function collectForm(blob, filename) {
  const fd = new FormData();
  fd.append("audio", blob, filename);
  fd.append("template_id", $("template").value);
  fd.append("doctor_id", $("doctor").value);
  const names = $("names").value.split(",").map((s) => s.trim()).filter(Boolean);
  fd.append("extra_names", JSON.stringify(names));
  const locked = {};
  $("locked").querySelectorAll("select[data-key]").forEach((s) => { if (s.value) locked[s.dataset.key] = s.value; });
  fd.append("locked", JSON.stringify(locked));
  localStorage.setItem("osce_doctor", $("doctor").value);
  localStorage.setItem("osce_template", $("template").value);
  return fd;
}

async function submit(blob, filename) {
  notice($("rec-msg"), "Mengunggah rekaman…");
  try {
    const res = await api("/api/jobs", { method: "POST", form: collectForm(blob, filename) });
    rememberJob(res.job_id);
    notice($("rec-msg"), `Job ${res.job_id} masuk antrean (posisi ${res.position}).`, "ok");
    refreshJobs();
  } catch (e) {
    notice($("rec-msg"), `Gagal mengunggah: ${e.message}`, "err");
  }
}

$("btn-start").onclick = async () => {
  try {
    recorder = new WavRecorder();
    await recorder.start();
    recorder.drawWaveform($("wave"));
    $("btn-start").disabled = true; $("btn-stop").disabled = false;
    $("rec-msg").replaceChildren();
    tick = setInterval(() => { $("timer").textContent = fmtTime(recorder.seconds); }, 250);
  } catch (e) {
    notice($("rec-msg"), e.message, "err");
  }
};

$("btn-stop").onclick = async () => {
  clearInterval(tick);
  $("btn-stop").disabled = true;
  const blob = await recorder.stop();
  const secs = recorder.seconds;
  recorder = null;
  $("btn-start").disabled = false;
  if (secs < 2) { notice($("rec-msg"), "Rekaman terlalu pendek.", "err"); return; }
  await submit(blob, `sesi_${Date.now()}.wav`);
};

$("btn-upload").onclick = async () => {
  const f = $("file").files[0];
  if (!f) { notice($("rec-msg"), "Pilih file audio terlebih dahulu.", "err"); return; }
  await submit(f, f.name);
};

$("template").onchange = renderLocked;

async function refreshJobs() {
  try {
    const q = await api("/api/queue");
    $("queue").textContent = `Sedang diproses: ${q.current || "—"} · menunggu: ${q.length} · VRAM: ${q.gpu_loaded.join(", ") || "kosong"}`;
    const ids = myJobs();
    const rows = await Promise.all(ids.map((id) => api(`/api/jobs/${id}?light=true`).catch(() => null)));
    $("jobs").replaceChildren(...rows.filter(Boolean).map((j) => el("tr", {},
      el("td", { class: "mono", text: j.id }),
      el("td", { text: j.template_id }),
      el("td", {}, el("span", { class: `pill ${j.status}`, text: j.status + (j.position > 0 ? ` #${j.position}` : "") })),
      el("td", { class: "muted", text: j.error ? j.error.slice(0, 80) : j.stage }),
      el("td", { class: "muted", text: j.created_at.replace("T", " ").slice(0, 19) }),
      el("td", {}, ["done", "verified"].includes(j.status) ? el("a", { href: `/review?job=${j.id}`, text: "Review →" }) : ""))));
  } catch (e) {
    $("queue").textContent = `Gagal memuat antrean: ${e.message}`;
  }
}

loadConfig().catch((e) => notice($("rec-msg"), e.message, "err"));
refreshJobs();
setInterval(refreshJobs, 3000);
