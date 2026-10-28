# TRIAGE
**Gated Evidence Selection and Context Decoupling for Clinical RAG**

A novel Retrieval-Augmented Generation framework that decomposes clinical semantics into orthogonal Entity–Relation–Modifier fields, applies graph-topology anisotropic transformations, and uses adaptive statistical Heaviside filtering combined with a calibrated JEV System One decision gate to mitigate context distraction and hallucinations in medical LLMs.

---

## Key Innovations

1. **Semantic Decoupling**: Represents each medical statement as a 3×d matrix $X_{\text{HCAD}} = [E; R; M]$ instead of a flat 1×d vector, decoupling entities, predicates, and modifiers.
2. **Anisotropic Transformation**: Computes a topology-aware transformation matrix $W_A$ to weight clinical embedding dimensions based on knowledge graph properties (node degree, relation confidence, IDF).
3. **Adaptive Heaviside Filtering**: Replaces rigid Top-K retrieval with a dual-gated dynamic threshold:
   $$\mathcal{M}_{\text{HCAD}}(t_i) = \Theta(\Phi_{\text{Final}}(t_i) - (\mu + k\sigma))$$
4. **JEV System One Decision Gate**: Employs an intelligent decision gate (`typesafe-sdk`) to evaluate factual groundedness and prune irrelevant or ungrounded context before prompt assembly.
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
       Clinical LLM ──► Response Synthesis
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Eklakh-AI-Engineer/Triage.git
cd Triage
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install SciSpacy biomedical NER models
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
```

### 4. Configure environment

Copy `.env.example` to `.env` and configure the provider settings you use. Do not commit API keys or secrets.

- **Local generation with Ollama:** set `LLM_PROVIDER=ollama`; ensure Ollama is installed and the configured models are available.
- **Optional experiment tracking:** configure Weights & Biases only if desired.
- **Optional remote decision gate / judge:** configure the required credentials explicitly and document the active mode for each experiment.

### 5. Prepare the knowledge corpus and indices

```bash
python src/main.py prepare --rebuild
```

### 6. Run a smoke test and benchmark

```bash
python src/main.py demo
python src/main.py benchmark --samples 50
```

Use the exact command options supported by `python src/main.py --help` for the current checkout. Large corpus preparation and full benchmarks can require substantial memory, disk space, GPU time, and model downloads.

---

## Evaluation

The repository includes a benchmark comparing:

1. Baseline LLM without retrieval
2. Vanilla dense vector RAG
3. HCAD-RAG / TRIAGE

Available evaluation code covers answer scoring, pairwise comparisons, retrieval quality, latency, citation-index validity, and MedHallu-related experiments. These metrics have different meanings and should not be interpreted interchangeably.

**Research status:** benchmark artifacts under `results/quarantine_invalid_vectorstore_2026-10-20/` were explicitly quarantined because they were produced with an incomplete vector store. Do not cite those numbers as valid performance results. Rebuild the corpus and run a fresh, fully documented evaluation before reporting results.

A reproducible run should record at least:

- Repository commit and dependency/model versions
- Dataset source, version, hashes, and indexed document counts
- Question-set identifier, split and random seed
- Generator, judge, prompts, and active gate mode
- Per-question outputs, failed judge calls, fallbacks, abstentions and latency
- Retrieval metrics and statistical analysis on a held-out test set

---

## Project Structure

```
Triage/
├── requirements.txt
├── README.md
├── data/
├── deploy/
├── models/
├── results/
└── src/
    ├── api/
    ├── baselines/
    ├── core/
    ├── data/
    ├── evaluation/
    ├── main.py
    └── wandb_tracking.py
```

---

## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| PrimeKG | [Harvard Dataverse](https://doi.org/10.7910/DVN/IXA7BM) | Biomedical knowledge graph |
| MedHallu | [Hugging Face](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | Medical hallucination benchmark |
| PMC Open Access | [NCBI / PMC](https://pmc.ncbi.nlm.nih.gov/) | Biomedical full-text corpus |
| HCAD-RAG | [Kaggle dataset](https://www.kaggle.com/datasets/nirajd071/hcad-rag) | Project-associated dataset; inspect dataset card for exact contents and version |

Review each dataset's licence and terms before redistribution.

---

## References

- **Hyper-RAG**: *Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation* ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758))
- **SIRG**: *Structured Information Retrieval Gating* ([arXiv:2601.03052](https://arxiv.org/abs/2601.03052))
- **PrimeKG**: Chandak et al., *Building a knowledge graph to enable precision medicine*, Scientific Data (2023)
- **SapBERT**: Liu et al., *Self-Alignment Pretraining for Biomedical Entity Representations*, NAACL (2021)

---

## License

Research and academic use only. Individual components are governed by their respective dataset and model licences.
