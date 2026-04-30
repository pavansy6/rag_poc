"""
Configuration Module
"""

# Generation model
MODEL_NAME = "qwen2.5:1.5b"

# Embedding model (used by Ollama / SentenceTransformer)
EMBEDDING_MODEL = "nomic-embed-text"

# FAISS index for your docs/ folder
FAISS_INDEX_PATH = "faiss_index"

# FAISS index built from MITRE ATT&CK JSON
MITRE_INDEX_PATH = "mitre_faiss_index"

# Path to the MITRE ATT&CK enterprise JSON bundle
# Download from: https://github.com/mitre/cti/raw/master/enterprise-attack/enterprise-attack.json
MITRE_JSON_PATH  = "data/enterprise-attack.json"

# docs/ folder
DOCS_PATH = "docs"

# Chunking settings (used by Chunker for docs/ only — MITRE has its own chunker)
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# System prompt — {context}, {query}, and optionally {history} are filled in by RAGEngine
SYSTEM_PROMPT = """
You are a Strict Internal Cybersecurity Auditor. Your sole purpose is to extract facts from the provided context.

### MANDATORY LOGIC RULES:
1. LITERAL EXTRACTION: You must only provide information explicitly stated in the ### CONTEXT ###.
2. NO INFERENCE: Do not use phrases like "this implies," "it is likely," or "suggests." If the text doesn't say it, it doesn't exist.
3. FALLBACK: If the answer is not found in the context, you must say: "I cannot find a specific rule or definition for this in the provided policy documents."
4. NO EXTERNAL DATA: Do not use your internal training data to explain security concepts (e.g., do not explain what MFA is unless the text explains it).
5. EXACT ATTRIBUTES: You must include exact numbers (SLAs, character lengths), technical standards (AES-256, TLS 1.3), and specific storage locations (Enterprise Vault) exactly as written.
6. ZERO FABRICATION: If the retrieved context does not contain the answer, you MUST use the fallback. Never substitute with general cybersecurity knowledge. Phishing is NOT credential dumping. Do not conflate topics.

### CONVERSATION HISTORY (if any):
{history}

### CONTEXT:
{context}

### QUESTION:
{query}
"""

# Per-model prompt templates. Keys match `prompt_template` values from the router mapping.
# PROMPT_TEMPLATES maps a prompt key (router's `prompt_template`) to a mapping
# of model_name -> template. If a model-specific template is not found,
# `get_prompt_template` falls back to the `default` entry or `SYSTEM_PROMPT`.
PROMPT_TEMPLATES = {
	"cyber": {
		"default": SYSTEM_PROMPT,
	},
	"ds": {
		# model-specific DS prompt for Llama
		"llama3.1:8b": """
You are a Data Science Assistant specialized for concise, reproducible answers.

CONVERSATION HISTORY (if any):
{history}

CONTEXT:
{context}

QUESTION:
{query}
""",
		"default": SYSTEM_PROMPT,
	},
	"general": {
		# model-specific general prompt for qwen
		"qwen2.5:1.5b": """
You are a knowledge assistant.

CONVERSATION HISTORY (if any):
{history}

CONTEXT:
{context}

QUESTION:
{query}
""",
		"default": SYSTEM_PROMPT,
	},
}


def get_prompt_template(prompt_key: str, model_name: str) -> str:
	"""Return the best prompt template for a given prompt_key and model_name.

	Lookup priority:
	1. PROMPT_TEMPLATES[prompt_key][model_name]
	2. PROMPT_TEMPLATES[prompt_key]["default"]
	3. SYSTEM_PROMPT
	"""
	pk = PROMPT_TEMPLATES.get(prompt_key, {})
	if model_name in pk:
		return pk[model_name]
	if "default" in pk:
		return pk["default"]
	return SYSTEM_PROMPT