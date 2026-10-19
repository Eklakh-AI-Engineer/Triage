# TRIAGE
**Gated Evidence Selection and Context Decoupling for Clinical RAG**

A novel Retrieval-Augmented Generation framework that decomposes clinical semantics into orthogonal Entity–Relation–Modifier fields, applies graph-topology anisotropic transformations, and uses adaptive statistical Heaviside filtering combined with a calibrated JEV System One decision gate to mitigate context distraction and hallucinations in medical LLMs.

---

## Key Innovations

1. **Semantic Decoupling**: Represents each medical statement as a 3×d matrix $X_{\text{HCAD}} = [E; R; M]$ instead of a flat 1×d vector, decoupling entities, predicates, and modifiers.
2. **Anisotropic Transformation**: Computes a topology-aware transformation matrix $W_A$ to weight clinical embedding dimensions based on knowledge graph properties (node degree, relation confidence, IDF).
3. **Adaptive Heaviside Filtering**: Replaces rigid Top-K retrieval with a dual-gated dynamic threshold:
   $$\mathcal{M}_{\text{HCAD}}(t_i) = \Theta(\Phi_{\text{Final}}(t_i) - (\mu + k\sigma))$$
4. **JEV System One Decision Gate**: Employs an intelligent decision gate (`typesafe-sdk`) to evaluate factual groundedness and prune irrelevant or ungrounded context before prompt assembly, preventing token bloat and hallucination injection.
5. **Bidirectional Hypergraph Expansion**: Discovers high-order biomedical relations and therapy associations from PrimeKG using multi-hop neighborhood hyperedges.

---

## Architecture

```
Clinical Query
      │
      ▼
Biomedical NER (SciSpacy / SapBERT) ──► Entity Linking (PrimeKG)
      │                                       │
      ▼                                       ▼
Relation & Modifier Extraction        Hyperedge Construction & Expansion
      │                                       │
      ▼                                       ▼
Semantic Decoupling (3×768 Matrix)    Candidate Context Pool
      │                                       │
      └──────────────────┬────────────────────┘
                         │
                         ▼
        Anisotropic Transformation & Tensor Scoring
                         │
                         ▼
       Adaptive Statistical Filter (Heaviside Gate)
                         │
                         ▼
       JEV System One Decision Gate (Groundedness Verification)
                         │
                         ▼
       Structured Clinical Prompt Construction
                         │
                         ▼
       Clinical LLM (Llama-3.1-8B-Instruct) ──► Verified Clinical Synthesis
```

---

## Quick Start

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/Nirajd071/HCAD-RAG.git
cd HCAD-RAG

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SciSpacy biomedical NER models
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
```

Use the repository-root `.env` as the single local environment file:
```bash
HF_TOKEN="your_huggingface_token"
TYPESAFE_API_KEY="your_typesafe_or_openrouter_api_key"  # Optional, enables live JEV System One gate
LLM_PROVIDER="huggingface"                              # "huggingface" or "ollama"
WANDB_API_KEY="your_rotated_wandb_api_key"              # Optional, enables experiment tracking
WANDB_PROJECT="hcad-rag"
# WANDB_ENTITY="your_wandb_username_or_team"            # Optional
```

The root `.env` is git-ignored and is loaded centrally by `src.config`; add
settings there rather than creating another environment file. A newly generated
W&B API key is loaded automatically for each evaluation, benchmark,
hallucination evaluation, and weight-learning run. Do not put API keys in source
code or commit them. Since a W&B key was shared in chat, revoke it in W&B and
replace it in `.env`. Set `WANDB_MODE="disabled"` to turn tracking off.

### 2. Prepare Knowledge Corpus & Indices

```bash
# Build knowledge graph indices and vector database with clinical relations
python src/data/build_knowledge_graph.py
python src/data/build_vector_store.py
```

### 3. Run Benchmark (Base Paper Protocol)

Replicate the evaluation protocol from Hyper-RAG ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758)):

```bash
# Run base paper benchmark comparing Baseline LLM, Vanilla RAG, and HCAD-RAG
python src/evaluation/run_base_paper_benchmark.py 4
```

Outputs:
- **5-Dimension Scoring Assessment** (0–100 scale: Comprehensiveness, Diversity, Empowerment, Logical, Readability)
- **8-Criteria Selection Assessment** (Pairwise Win-Rate with order counterbalancing)
- **Retrieval Latency Profiling**
- MedQA accuracy is reported with a 95% Wilson interval; benchmark system differences include paired bootstrap 95% intervals over the same sampled questions. Sampling uses a local seeded RNG, so it is reproducible without changing global random state.
- W&B aggregate and per-question numeric MedQA metrics plus MedHallu scalar metrics when `WANDB_API_KEY` is configured. MedQA and MedHallu charts are not logged; base-paper scoring, selection, and latency comparison charts remain enabled. Logged MedQA sample tables omit question and response text.

Weight learning tunes only `filter_k`, the learned parameter currently consumed by
the retriever. Anisotropic and component-scoring weights remain at their configured
values; retrieval and model architecture are unchanged.

---

## Project Structure

```
hcad-rag/
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation
├── .gitignore                         # Git exclusion rules
├── src/
│   ├── config.py                      # Global configurations, model weights & paths
│   ├── main.py                        # CLI entry point
│   ├── data/
│   │   ├── download_datasets.py       # Dataset fetchers (PrimeKG, MedQA, MedHallu)
│   │   ├── build_knowledge_graph.py   # PrimeKG MultiDiGraph & index construction
│   │   └── build_vector_store.py      # ChromaDB clinical context ingestion
│   ├── core/
│   │   ├── ner_extractor.py           # SciSpacy + Regex Biomedical NER
│   │   ├── entity_linker.py           # SapBERT FAISS cosine entity resolution
│   │   ├── relation_extractor.py      # Rule-based clinical predicate extraction
│   │   ├── semantic_decoupler.py      # Innovation 1: Entity-Relation-Modifier decoupling
│   │   ├── anisotropic_transformer.py # Innovation 2: Graph topology W_A transformation
│   │   ├── hypergraph_builder.py      # Innovation 3: Bidirectional hyperedge expansion
│   │   ├── statistical_filter.py      # Innovation 4: Adaptive Heaviside cutoff
│   │   ├── decision_gate.py           # Innovation 5: JEV System One groundedness gate
│   │   ├── prompt_builder.py          # Standardized 4-section clinical prompt builder
│   │   ├── llm_client.py              # Unified LLM interface (HuggingFace / Ollama)
│   │   └── hcad_retriever.py          # End-to-end integrated HCAD-RAG pipeline
│   ├── baselines/
│   │   ├── baseline_llm.py            # Parametric LLM baseline (zero-retrieval)
│   │   └── vanilla_vector_rag.py      # Standard dense Top-K vector baseline
│   └── evaluation/
│       ├── base_paper_evaluator.py    # Scoring & selection rubric evaluator (LLM-as-a-judge)
│       ├── run_base_paper_benchmark.py# Base paper benchmark test runner
│       ├── medqa_evaluator.py         # USMLE MCQ evaluation
│       └── results_analyzer.py        # Results visualization & table export
└── results/
    ├── figures/                       # Benchmark charts
    └── tables/                        # Summary metrics & logs
```

---

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| [PrimeKG](https://doi.org/10.7910/DVN/IXA7BM) | Harvard Dataverse | Precision medicine knowledge graph (129K nodes, 4M+ edges) |
| [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) | HuggingFace | Standard clinical board question benchmark (USMLE 4-option) |
| [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | HuggingFace | Medical hallucination detection dataset |
| PMC Open Access | NCBI / PMC | Curated open-access clinical text corpus |

---

## References

- **Hyper-RAG**: *Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation* ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758))
- **SIRG**: *Structured Information Retrieval Gating* ([arXiv:2601.03052](https://arxiv.org/abs/2601.03052))
- **PrimeKG**: Chandak et al., *Building a knowledge graph to enable precision medicine*, Scientific Data (2023)
- **SapBERT**: Liu et al., *Self-Alignment Pretraining for Biomedical Entity Representations*, NAACL (2021)

---

## License

Research and academic use only. Individual components are governed by their respective dataset and model licenses.
