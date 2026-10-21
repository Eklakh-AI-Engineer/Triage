import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
TENSOR_MODEL_PATH = MODELS_DIR / "dynamic_tensor.pt"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORDB_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_FILENAME = "kg.csv"

MEDHALLU_DATASET_ID = "UTAustin-AIHealth/MedHallu"
MEDHALLU_CONFIG_LABELED = "pqa_labeled"
MEDHALLU_CONFIG_ARTIFICIAL = "pqa_artificial"

# Embedding & Entity Linking Models
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768
ENTITY_LINKING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# HCAD Core Parameters
NUM_SEMANTIC_COMPONENTS = 3  # Entity, Relation, Modifier
HCAD_MATRIX_SHAPE = (NUM_SEMANTIC_COMPONENTS, EMBEDDING_DIMENSION)

# Anisotropic Transformation Weights
ANISO_ALPHA = 0.4   # Node degree
ANISO_BETA = 0.4    # Edge confidence
ANISO_GAMMA = 0.2   # Inverse document frequency

# Retrieval Score Weights
SCORE_WEIGHT_ENTITY = 0.3
SCORE_WEIGHT_RELATION = 0.5
SCORE_WEIGHT_MODIFIER = 0.2

# Statistical Filtering
FILTERING_K = 1.0
MIN_PHI_SCORE = 0.10

# Semantic Decoupling 
ORTHO_STRENGTH = 0.70        # soft Gram-Schmidt blend weight (eta)
GS_EPS = 1e-8
GS_RESIDUAL_TOL = 1e-3      # below this, a component is marked absent (presence mask)
NEUTRAL_RELATION_TEXT = "biomedical relation"   # anchor, never equal to entity text
NEUTRAL_MODIFIER_TEXT = "no qualifier stated"

# Knowledge-graph relation exclusion policy 
KG_EXCLUDED_RELATIONS = {"drug_drug"}

# LLM
LLM_OLLAMA_TIMEOUT_S = 120.0

# Graph Expansion
GRAPH_HOP_DEPTH = 2
MAX_NEIGHBORS_PER_HOP = 50
MAX_EVIDENCE_ITEMS = 8
EVIDENCE_WORD_BUDGET = 350

# Vanilla Baseline
VANILLA_TOP_K = 5

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")  # "nvidia", "huggingface", or "ollama"
LLM_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Separate judge to avoid self-preference bias
# Base-paper parity : 72B-class judge as in the paper's
# protocol. Fits A100 80GB locally (Q4). Override to qwen3:14b for small GPUs.
JUDGE_OLLAMA_MODEL = os.getenv("JUDGE_OLLAMA_MODEL", "qwen2.5:72b")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Text-optimized NIM models (vision-instruct variants underperform on clinical
# prose and waste tokens on the image tower). Override via env if needed.
LLM_NVIDIA_MODEL = os.getenv("LLM_NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
JUDGE_NVIDIA_MODEL = os.getenv("JUDGE_NVIDIA_MODEL", "qwen/qwen2.5-72b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hub downloads (SapBERT/S-PubMedBert) still need this
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "hcad-rag")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None
WANDB_MODE = os.getenv("WANDB_MODE", "online")
LLM_MODEL = "llama3.1:8b"
# Base-paper Fig. 3 parity: evaluate augmentation generality across generators.
BENCHMARK_GENERATORS = [LLM_MODEL]  # e.g. ["llama3.1:8b", "qwen3:14b", "deepseek-r1:8b"]
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
ENABLE_FEW_SHOT = False  # zero-shot default; opt into few-shot explicitly
FEW_SHOT_COUNT = 4  # consumed by PromptBuilder.build_mcq_prompt

# Embeddings are always L2-normalized at encode time (cosine consistency)

LLM_SYSTEM_PROMPT = """You are a biomedical research assistant specializing in clinical medicine.
Use the provided context as primary evidence, and synthesize it with your expert clinical
domain knowledge to deliver comprehensive, accurate, and actionable answers.
Clearly cite specific evidence from the context where applicable, and supplement
with established medical knowledge when the context is incomplete.
Be precise, thorough, and clinically rigorous."""

# Vector Database
CHROMA_COLLECTION_NAME = "biomedical_contexts"
CHROMA_DISTANCE_METRIC = "cosine"

# NER Models
NER_MODELS = {
    "drug_disease": "en_ner_bc5cdr_md",
    "molecular": "en_ner_jnlpba_md",
}

# Relation Types & Negation
RELATION_TYPES = [
    "TREATS",
    "CAUSES",
    "CONTRAINDICATED",
    "INTERACTS_WITH",
    "TARGETS",
    "ASSOCIATED_WITH",
    "INHIBITS",
    "ACTIVATES",
    "UPREGULATES",
    "DOWNREGULATES",
    "BIOMARKER_FOR",
    "NOT_RECOMMENDED",
    "INDICATED_FOR",
]

NEGATION_CUES = [
    "not", "no", "never", "without", "avoid", "contraindicated",
    "should not", "must not", "do not", "doesn't", "does not",
    "isn't", "is not", "wasn't", "was not", "cannot", "can't",
    "not recommended", "discontinued", "withdrawn", "prohibited",
    "lack of", "absence of", "failure to",
    "rules out", "ruled out", "unlikely", "fails to improve", "cannot tolerate",
    "refractory to", "excluded",
]

# Evaluation Parameters
DEFAULT_EVAL_SAMPLES = 200
RANDOM_SEED = 42

