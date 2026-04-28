"""Utilities for parsing MITRE ATT&CK JSON into retrievable text chunks.

This module loads MITRE ATT&CK content from the official JSON format and
converts it into text chunks with metadata for indexing.
"""

import json
import re
from pathlib import Path
from collections import Counter

def _clean(text: str) -> str:
    """Remove MITRE citation markers and trim whitespace from text."""
    return re.sub('\\(Citation:[^)]+\\)', '', text or '').strip()

def _mitre_id(obj: dict) -> str:
    """Extract the MITRE ATT&CK external ID from an object."""
    for ref in obj.get('external_references', []):
        if ref.get('source_name') == 'mitre-attack':
            return ref.get('external_id', '')
    return ''

def _tactic_names(obj: dict) -> list[str]:
    """Return the MITRE ATT&CK tactic phase names for an object."""
    return [p['phase_name'] for p in obj.get('kill_chain_phases', []) if p.get('kill_chain_name') == 'mitre-attack']

def _platforms(obj: dict) -> list[str]:
    """Normalize platform data from a MITRE object.

    Some ATT&CK entries store platforms as strings and others as dicts.
    """
    return [p if isinstance(p, str) else p.get('platform_name', '') for p in obj.get('x_mitre_platforms', [])]

def load_mitre_documents(json_path: str) -> list[dict]:
    """Load MITRE ATT&CK JSON and convert objects into searchable text chunks.

    Args:
        json_path (str): Path to the MITRE ATT&CK JSON bundle.

    Returns:
        list[dict]: Chunk dictionaries containing ``text`` and ``metadata``.
    """
    objects = json.loads(Path(json_path).read_text(encoding='utf-8')).get('objects', [])
    tactics = {}
    techniques = []
    mitigations = {}
    relationships = []
    
    for obj in objects:
        t = obj.get('type', '')
        if t == 'x-mitre-tactic':
            tactics[obj['x_mitre_shortname']] = obj
        elif t == 'attack-pattern':
            techniques.append(obj)
        elif t == 'course-of-action':
            mitigations[obj['id']] = obj
        elif t == 'relationship':
            relationships.append(obj)
    mit_lookup: dict[str, list[str]] = {}
    
    for rel in relationships:
        if rel.get('relationship_type') == 'mitigates':
            target = rel.get('target_ref', '')
            source = mitigations.get(rel.get('source_ref', ''))
            if source:
                mit_lookup.setdefault(target, []).append(source.get('name', ''))
    docs = []
    
    for shortname, tactic in tactics.items():
        tactic_id = _mitre_id(tactic)
        tactic_name = tactic.get('name', '')
        child_ids = [_mitre_id(t) for t in techniques if shortname in _tactic_names(t) and (not t.get('x_mitre_is_subtechnique')) and (not t.get('revoked')) and (not t.get('x_mitre_deprecated'))]
        text = f"Tactic: {tactic_name} ({tactic_id})\n{_clean(tactic.get('description', ''))}\nTechniques: {', '.join(child_ids)}"
        docs.append({'text': text, 'metadata': {'chunk_type': 'tactic', 'tactic_id': tactic_id, 'tactic_name': tactic_name, 'technique_ids': child_ids}})
    
    for tech in techniques:
        
        if tech.get('revoked') or tech.get('x_mitre_deprecated'):
            continue
        
        tech_id = _mitre_id(tech)
        tech_name = tech.get('name', '')
        is_sub = bool(tech.get('x_mitre_is_subtechnique'))
        tactic_list = _tactic_names(tech)
        description = _clean(tech.get('description', ''))
        detection = _clean(tech.get('x_mitre_detection', ''))
        
        mits = mit_lookup.get(tech['id'], [])
        parts = [f"{('Sub-technique' if is_sub else 'Technique')}: {tech_name} ({tech_id})", f"Tactic(s): {', '.join(tactic_list)}", description]
        
        if detection:
            parts.append(f'Detection: {detection}')
        if mits:
            parts.append(f"Mitigations: {', '.join(mits)}")
        
        text = '\n'.join(parts)
        metadata = {'chunk_type': 'sub-technique' if is_sub else 'technique', 'technique_id': tech_id, 'technique_name': tech_name, 'tactic_names': tactic_list, 'platforms': _platforms(tech), 'is_subtechnique': is_sub}
        
        if is_sub:
            metadata['parent_technique_id'] = tech_id.rsplit('.', 1)[0]
        
        docs.append({'text': text, 'metadata': metadata})
    
    print(f"[mitre_chunker] {len(docs)} chunks — {sum((1 for d in docs if d['metadata']['chunk_type'] == 'tactic'))} tactics, {sum((1 for d in docs if d['metadata']['chunk_type'] == 'technique'))} techniques, {sum((1 for d in docs if d['metadata']['chunk_type'] == 'sub-technique'))} sub-techniques")
    return docs

def save_mitre_documents(texts: list[str], meta: list[dict], output_path: str = "mitre_chunks.json"):
    """Save parsed MITRE chunks and summary metadata to a JSON file."""
    counts = Counter(m["chunk_type"] for m in meta)
    output = {
        "summary": {
            "total":          len(texts),
            "tactics":        counts.get("tactic", 0),
            "techniques":     counts.get("technique", 0),
            "sub_techniques": counts.get("sub-technique", 0),
        },
        "chunks": [{"text": t, "metadata": m} for t, m in zip(texts, meta)]
    }
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[mitre_chunker] Saved {len(texts)} chunks → {output_path}")
