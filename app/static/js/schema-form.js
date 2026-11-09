// Render form dari JSON Schema (hasil inline_json_schema Pydantic) -> editor rangkuman terstandar.
// Mendukung: enum (dropdown terkunci), string, boolean, array enum (checkbox), array EvidenceItem, object.
import { el } from "./api.js";

const humanize = (k) => k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

function enumOf(schema) {
  if (schema.enum) return schema.enum;
  if (schema.anyOf) { const e = schema.anyOf.find((s) => s.enum); if (e) return e.enum; }
  return null;
}

function isEvidenceArray(schema) {
  return schema.type === "array" && schema.items && schema.items.type === "object" &&
    schema.items.properties && "teks" in schema.items.properties && "sumber" in schema.items.properties;
}

export function renderSchemaForm(schema, value, { onSourceClick, readOnly = false } = {}) {
  const state = structuredClone(value || {});

  function field(key, sch, holder) {
    const title = sch.title || humanize(key);
    const opts = enumOf(sch);
    if (opts) {
      const sel = el("select", { disabled: readOnly, onchange: (e) => { holder[key] = e.target.value; } },
        opts.map((o) => el("option", { value: o, text: o, selected: holder[key] === o })));
      if (holder[key] === undefined) holder[key] = sch.default ?? opts[0];
      return el("div", {}, el("label", { text: title }), sel);
    }
    if (sch.type === "boolean") {
      if (holder[key] === undefined) holder[key] = !!sch.default;
      return el("div", { class: "checks" }, el("label", {},
        el("input", { type: "checkbox", checked: !!holder[key], disabled: readOnly,
          onchange: (e) => { holder[key] = e.target.checked; } }), title));
    }
    if (sch.type === "string") {
      if (holder[key] === undefined) holder[key] = sch.default ?? "";
      return el("div", {}, el("label", { text: title }),
        el("textarea", { rows: 1, value: holder[key], readOnly, oninput: (e) => { holder[key] = e.target.value; } }));
    }
    if (sch.type === "array" && sch.items && enumOf(sch.items)) {
      if (!Array.isArray(holder[key])) holder[key] = [];
      return el("div", {}, el("label", { text: title }), el("div", { class: "checks" },
        enumOf(sch.items).map((o) => el("label", {}, el("input", {
          type: "checkbox", checked: holder[key].includes(o), disabled: readOnly,
          onchange: (e) => {
            holder[key] = e.target.checked ? [...new Set([...holder[key], o])] : holder[key].filter((x) => x !== o);
          },
        }), o))));
    }
    if (isEvidenceArray(sch)) return evidenceList(key, title, holder);
    if (sch.type === "object" && sch.properties) {
      if (typeof holder[key] !== "object" || holder[key] === null) holder[key] = {};
      return el("fieldset", {}, el("legend", { text: title }),
        Object.entries(sch.properties).map(([k, s]) => field(k, s, holder[key])));
    }
    return el("div", { class: "muted", text: `${title}: tipe tidak didukung editor` });
  }

  function evidenceList(key, title, holder) {
    if (!Array.isArray(holder[key])) holder[key] = [];
    const list = el("div");
    const redraw = () => {
      list.replaceChildren(...holder[key].map((item, idx) => {
        const chips = el("div", {}, (item.sumber || []).map((id) =>
          el("span", { class: "chip", text: `#${id}`, title: "Lihat utterance", onclick: () => onSourceClick && onSourceClick(id) })));
        return el("div", { class: "ev-item" },
          el("textarea", { rows: 2, value: item.teks, readOnly, oninput: (e) => { item.teks = e.target.value; } }),
          el("div", {},
            el("input", { type: "text", value: (item.sumber || []).join(","), placeholder: "id sumber", readOnly,
              onchange: (e) => {
                item.sumber = e.target.value.split(/[,\s]+/).filter(Boolean).map(Number).filter(Number.isInteger);
                redraw();
              } }), chips),
          readOnly ? el("span") : el("button", { class: "small", text: "✕", title: "Hapus butir",
            onclick: () => { holder[key].splice(idx, 1); redraw(); } }));
      }));
    };
    redraw();
    return el("fieldset", {}, el("legend", { text: title }), list,
      readOnly ? null : el("button", { class: "small", text: "+ butir", onclick: () => { holder[key].push({ teks: "", sumber: [] }); redraw(); } }));
  }

  const root = el("div", {}, Object.entries(schema.properties || {}).map(([k, s]) => field(k, s, state)));
  return { element: root, getValue: () => structuredClone(state) };
}
