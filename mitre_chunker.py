"""
MITRE ATT&CK → LangChain Documents for FAISS
─────────────────────────────────────────────
Usage:
    from mitre_chunker import load_mitre_documents
    docs = load_mitre_documents("enterprise-attack.json")

    # Drop straight into FAISS
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings          # or your embeddings

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("mitre_faiss_index")
"""

import json
import re
from pathlib import Path
from typing import Optional
from langchain_core.documents import Document


# ─── helpers ────────────────────────────────────────────────────────────────

def _plain(text: Optional[str]) -> str:
    """Strip citation references like (Citation: ...) from descriptions."""
    if not text:
        return ""
    return re.sub(r"\(Citation:[^)]+\)", "", text).strip()


def _platforms(obj: dict) -> list[str]:
    return [
        p if isinstance(p, str) else p.get("platform_name", "")
        for p in obj.get("x_mitre_platforms", [])
    ]


def _data_sources(obj: dict) -> list[str]:
    return obj.get("x_mitre_data_sources", [])


def _get_kill_chain_phases(obj: dict) -> list[dict]:
    return obj.get("kill_chain_phases", [])


def _phase_names(obj: dict) -> list[str]:
    return [p["phase_name"] for p in _get_kill_chain_phases(obj) if p.get("kill_chain_name") == "mitre-attack"]


def _short_id(obj: dict) -> str:
    """Extract T1234 / T1234.001 / TA0001 from external_references."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id", "")
    return ""


def _mitigations_text(relationships: list[dict], mitigations_by_id: dict, technique_stix_id: str) -> str:
    """Build a short mitigation summary for a technique."""
    lines = []
    for rel in relationships:
        if (
            rel.get("relationship_type") == "mitigates"
            and rel.get("target_ref") == technique_stix_id
        ):
            mit = mitigations_by_id.get(rel["source_ref"])
            if mit:
                name = mit.get("name", "")
                desc = _plain(mit.get("description", ""))[:200]
                lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


# ─── main parser ────────────────────────────────────────────────────────────

def load_mitre_documents(json_path: str | Path) -> list[Document]:
    """
    Parse the MITRE ATT&CK enterprise JSON bundle and return a flat list of
    LangChain Document objects — one per tactic, technique, and sub-technique.

    Each Document has:
        .page_content  – the text that gets embedded
        .metadata      – structured fields for metadata filtering
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    objects = raw.get("objects", [])

    # ── index objects by STIX id ──────────────────────────────────────────
    tactics_by_shortname: dict[str, dict] = {}   # phase_name → x-mitre-tactic obj
    techniques: list[dict] = []
    mitigations_by_id: dict[str, dict] = {}
    relationships: list[dict] = []

    for obj in objects:
        t = obj.get("type", "")
        if t == "x-mitre-tactic":
            tactics_by_shortname[obj["x_mitre_shortname"]] = obj
        elif t == "attack-pattern":
            techniques.append(obj)
        elif t == "course-of-action":
            mitigations_by_id[obj["id"]] = obj
        elif t == "relationship":
            relationships.append(obj)

    docs: list[Document] = []

    # ── 1. TACTIC documents ──────────────────────────────────────────────
    for shortname, tactic in tactics_by_shortname.items():
        tactic_id   = _short_id(tactic)
        tactic_name = tactic.get("name", "")
        description = _plain(tactic.get("description", ""))

        # collect technique IDs that belong to this tactic
        child_ids = [
            _short_id(t)
            for t in techniques
            if shortname in _phase_names(t) and not t.get("x_mitre_is_subtechnique")
        ]

        page_content = (
            f"Tactic: {tactic_name} ({tactic_id})\n"
            f"{description}\n"
            f"Techniques in this tactic: {', '.join(child_ids)}"
        )

        docs.append(Document(
            page_content=page_content,
            metadata={
                "chunk_type":   "tactic",
                "tactic_id":    tactic_id,
                "tactic_name":  tactic_name,
                "tactic_shortname": shortname,
                "technique_ids": child_ids,
            }
        ))

    # ── 2. TECHNIQUE + SUB-TECHNIQUE documents ────────────────────────────
    for tech in techniques:
        if tech.get("x_mitre_deprecated") or tech.get("revoked"):
            continue

        tech_id     = _short_id(tech)
        tech_name   = tech.get("name", "")
        description = _plain(tech.get("description", ""))
        detection   = _plain(tech.get("x_mitre_detection", ""))
        platforms   = _platforms(tech)
        data_srcs   = _data_sources(tech)
        tactic_names = _phase_names(tech)

        # map phase_name back to tactic_id
        tactic_ids = [
            _short_id(tactics_by_shortname[p])
            for p in tactic_names
            if p in tactics_by_shortname
        ]

        is_sub = bool(tech.get("x_mitre_is_subtechnique"))

        # mitigation summary (only for top-level techniques; subs inherit parent's)
        mit_text = ""
        if not is_sub:
            mit_text = _mitigations_text(relationships, mitigations_by_id, tech["id"])

        # ── build page_content ──────────────────────────────────────────
        parts = [
            f"{'Sub-technique' if is_sub else 'Technique'}: {tech_name} ({tech_id})",
            f"Tactic(s): {', '.join(tactic_names)}",
            "",
            description,
        ]
        if detection:
            parts += ["", "Detection:", detection]
        if mit_text:
            parts += ["", "Mitigations:", mit_text]

        page_content = "\n".join(parts)

        # ── metadata ────────────────────────────────────────────────────
        metadata: dict = {
            "chunk_type":     "sub-technique" if is_sub else "technique",
            "technique_id":   tech_id,
            "technique_name": tech_name,
            "tactic_ids":     tactic_ids,
            "tactic_names":   tactic_names,
            "platforms":      platforms,
            "data_sources":   data_srcs,
            "is_subtechnique": is_sub,
        }

        if is_sub:
            # e.g. T1566.001 → parent is T1566
            parent_id = tech_id.rsplit(".", 1)[0] if "." in tech_id else ""
            metadata["parent_technique_id"] = parent_id
        else:
            # attach mitigation IDs for later filtering
            mit_ids = [
                rel["source_ref"]
                for rel in relationships
                if rel.get("relationship_type") == "mitigates"
                and rel.get("target_ref") == tech["id"]
            ]
            metadata["mitigation_ids"] = mit_ids

        docs.append(Document(page_content=page_content, metadata=metadata))

    print(f"[mitre_chunker] Parsed {len(docs)} documents total")
    _print_summary(docs)
    return docs


def _print_summary(docs: list[Document]):
    from collections import Counter
    counts = Counter(d.metadata["chunk_type"] for d in docs)
    for k, v in sorted(counts.items()):
        print(f"  {k:<20} {v} docs")



if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "enterprise-attack.json"
    docs = load_mitre_documents(path)

    # preview first doc of each type
    seen = set()
    for doc in docs:
        ct = doc.metadata["chunk_type"]
        if ct not in seen:
            seen.add(ct)
            print(f"\n{'─'*60}")
            print(f"chunk_type: {ct}")
            print(f"metadata:   {doc.metadata}")
            print(f"content:\n{doc.page_content[:400]}...")