"""Template standar penulisan klinis.

Semua field kategorikal memakai `Literal` -> JSON Schema `enum` -> grammar constrained decoding,
sehingga LLM tidak bisa mengeluarkan nilai di luar opsi. Setiap butir naratif wajib menyertakan
`sumber` (ID utterance) agar klaim bisa ditelusuri balik ke transkrip (anti-halusinasi + HITL).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

TD = "Tidak disebutkan"

Onset = Literal["Akut (<2 minggu)", "Subakut (2-12 minggu)", "Kronis (>12 minggu)", "Tidak disebutkan"]
SkalaNyeri = Literal["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Tidak disebutkan"]
SistemOrgan = Literal["Respirasi", "Kardiovaskular", "Gastrointestinal", "Neurologi", "Muskuloskeletal",
                      "Urogenital", "Integumen", "Endokrin-Metabolik", "THT", "Mata", "Psikiatri",
                      "Lainnya", "Tidak disebutkan"]
YaTidak = Literal["Ya", "Tidak", "Tidak disebutkan"]


class EvidenceItem(BaseModel):
    teks: str = Field(description="Kalimat ringkas berbahasa medis baku.")
    sumber: list[int] = Field(default_factory=list, description="ID utterance pendukung.")


class SoapUmum(BaseModel):
    keluhan_utama: str = Field(default=TD)
    onset: Onset = TD
    skala_nyeri: SkalaNyeri = TD
    sistem_organ: SistemOrgan = TD
    subjective: list[EvidenceItem] = Field(default_factory=list)
    objective: list[EvidenceItem] = Field(default_factory=list)
    assessment: list[EvidenceItem] = Field(default_factory=list)
    plan: list[EvidenceItem] = Field(default_factory=list)


class CalgaryCambridge(BaseModel):
    keluhan_utama: str = Field(default=TD)
    onset: Onset = TD
    sistem_organ: SistemOrgan = TD
    riwayat_penyakit_sekarang: list[EvidenceItem] = Field(default_factory=list)
    riwayat_penyakit_dahulu: list[EvidenceItem] = Field(default_factory=list)
    riwayat_pengobatan: list[EvidenceItem] = Field(default_factory=list)
    riwayat_alergi: list[EvidenceItem] = Field(default_factory=list)
    riwayat_keluarga: list[EvidenceItem] = Field(default_factory=list)
    riwayat_sosial_ekonomi: list[EvidenceItem] = Field(default_factory=list)
    ide_pasien: list[EvidenceItem] = Field(default_factory=list)
    kekhawatiran_pasien: list[EvidenceItem] = Field(default_factory=list)
    harapan_pasien: list[EvidenceItem] = Field(default_factory=list)


class RubrikRespirasi(BaseModel):
    menanyakan_onset_dan_durasi: bool = False
    menanyakan_karakter_batuk: bool = False
    menanyakan_sesak_napas: bool = False
    menanyakan_demam: bool = False
    menanyakan_riwayat_merokok: bool = False
    menanyakan_kontak_tb: bool = False
    menanyakan_penurunan_berat_badan: bool = False
    menanyakan_keringat_malam: bool = False
    menyampaikan_diagnosis: bool = False
    memberikan_edukasi: bool = False


class OsceRespirasi(SoapUmum):
    batuk: Literal["Tidak ada", "Kering", "Berdahak", "Berdarah (hemoptisis)", "Tidak disebutkan"] = TD
    sesak_napas: YaTidak = TD
    demam: YaTidak = TD
    riwayat_merokok: Literal["Perokok aktif", "Mantan perokok", "Bukan perokok", "Tidak disebutkan"] = TD
    kontak_tb: YaTidak = TD
    rubrik: RubrikRespirasi = Field(default_factory=RubrikRespirasi)


FaktorRisikoKV = Literal["Hipertensi", "Diabetes melitus", "Dislipidemia", "Merokok",
                         "Riwayat keluarga PJK", "Obesitas"]


class RubrikKardiovaskular(BaseModel):
    menanyakan_lokasi_nyeri: bool = False
    menanyakan_karakter_nyeri: bool = False
    menanyakan_penjalaran: bool = False
    menanyakan_durasi_nyeri: bool = False
    menanyakan_faktor_pencetus: bool = False
    menanyakan_gejala_penyerta: bool = False
    menanyakan_faktor_risiko: bool = False
    menyampaikan_diagnosis: bool = False
    memberikan_edukasi: bool = False


class OsceKardiovaskular(SoapUmum):
    nyeri_dada: YaTidak = TD
    karakter_nyeri: Literal["Tertekan/berat", "Tertusuk", "Terbakar", "Tidak disebutkan"] = TD
    penjalaran: Literal["Lengan kiri", "Rahang/leher", "Punggung", "Epigastrium", "Tidak menjalar",
                        "Tidak disebutkan"] = TD
    sesak_napas: YaTidak = TD
    palpitasi: YaTidak = TD
    edema_tungkai: YaTidak = TD
    faktor_risiko: list[FaktorRisikoKV] = Field(default_factory=list)
    rubrik: RubrikKardiovaskular = Field(default_factory=RubrikKardiovaskular)


@dataclass(frozen=True)
class TemplateSpec:
    id: str
    title: str
    description: str
    model: type[BaseModel]
    # bagaimana fallback rule-based memetakan kelas intent -> field list
    intent_field_map: dict[str, str]


TEMPLATES: dict[str, TemplateSpec] = {
    "soap_umum": TemplateSpec(
        "soap_umum", "Standar SOAP Umum",
        "Catatan SOAP umum rawat jalan.",
        SoapUmum,
        {"S_KELUHAN": "subjective", "O_PEMERIKSAAN": "objective",
         "A_DIAGNOSIS": "assessment", "P_EDUKASI_TERAPI": "plan"},
    ),
    "calgary_cambridge": TemplateSpec(
        "calgary_cambridge", "Anamnesis Calgary-Cambridge",
        "Struktur anamnesis Calgary-Cambridge termasuk ICE (ideas, concerns, expectations).",
        CalgaryCambridge,
        {"S_KELUHAN": "riwayat_penyakit_sekarang"},
    ),
    "osce_respirasi": TemplateSpec(
        "osce_respirasi", "Rubrik OSCE Sistem Respirasi",
        "SOAP + parameter terkunci dan checklist rubrik anamnesis respirasi.",
        OsceRespirasi,
        {"S_KELUHAN": "subjective", "O_PEMERIKSAAN": "objective",
         "A_DIAGNOSIS": "assessment", "P_EDUKASI_TERAPI": "plan"},
    ),
    "osce_kardiovaskular": TemplateSpec(
        "osce_kardiovaskular", "Rubrik OSCE Sistem Kardiovaskular",
        "SOAP + parameter terkunci dan checklist rubrik anamnesis kardiovaskular.",
        OsceKardiovaskular,
        {"S_KELUHAN": "subjective", "O_PEMERIKSAAN": "objective",
         "A_DIAGNOSIS": "assessment", "P_EDUKASI_TERAPI": "plan"},
    ),
}


def get_template(template_id: str) -> TemplateSpec:
    try:
        return TEMPLATES[template_id]
    except KeyError as exc:
        raise ValueError(f"Template tidak dikenal: {template_id!r}. Pilihan: {list(TEMPLATES)}") from exc


def inline_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema tanpa $ref (beberapa konverter schema->grammar tidak mendukung $ref penuh)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 20:
            raise ValueError("Schema terlalu dalam / rekursif.")
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = copy.deepcopy(defs[name])
                extra = {k: v for k, v in node.items() if k != "$ref"}
                merged.update(extra)
                return resolve(merged, depth + 1)
            return {k: resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v, depth + 1) for v in node]
        return node

    return resolve(schema)


def llm_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Schema untuk constrained decoding LLM: seperti inline_json_schema, tetapi SEMUA properti object
    wajib (`required`). Tanpa ini, field ber-default bersifat opsional dan grammar mengizinkan LLM
    mengeluarkan object minimal (mis. hanya `rubrik`), lalu sisanya terisi default "Tidak disebutkan".
    """
    def require_all(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: require_all(v) for k, v in node.items()}
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            return node
        if isinstance(node, list):
            return [require_all(v) for v in node]
        return node

    return require_all(inline_json_schema(model))


def evidence_fields(model: type[BaseModel]) -> list[str]:
    """Nama field bertipe list[EvidenceItem]."""
    out: list[str] = []
    for name, field in model.model_fields.items():
        ann = field.annotation
        args = getattr(ann, "__args__", ())
        if args and args[0] is EvidenceItem:
            out.append(name)
    return out
