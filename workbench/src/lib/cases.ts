import type {
  AnswerSegment,
  BenchmarkResponse,
  CitationRecord,
  EntityType,
  GraphEdge,
  GraphNode,
  SubgraphResponse,
} from "./types";

// ---------------------------------------------------------------------------
// Standalone mock substrate. Medically grounded on PrimeKG-style relations
// (contraindication / drug_effect / indication / carrier / ...) and PMC
// guideline abstracts. Replaced by live backend responses once
// TRIAGE_API_BASE is configured.
// ---------------------------------------------------------------------------

export interface ClinicalCase {
  id: string;
  label: string;
  query: string;
  keywords: string[];
  entities: { name: string; type: EntityType; nodeId?: string }[];
  candidates: { text: string; score: number; source: "PrimeKG" | "PMC"; redundancy: number }[];
  citations: Record<string, CitationRecord>;
  subgraph: SubgraphResponse;
  arena: {
    baseline: AnswerSegment[];
    vanillaRag: AnswerSegment[];
    triage: AnswerSegment[];
    warnings: string[];
  };
}

const metforminCitations: Record<string, CitationRecord> = {
  "PrimeKG:#8921": {
    id: "PrimeKG:#8921",
    triple: "Metformin —[contraindication]→ Chronic Kidney Disease (eGFR < 30 mL/min/1.73m²)",
    source: "PrimeKG relation: contraindication (drug_contraindicates_disease)",
    confidence: 0.94,
    abstract:
      "Metformin is contraindicated in severe renal impairment (eGFR below 30 mL/min/1.73m²). Reduced renal clearance leads to drug accumulation and elevated lactate production risk.",
  },
  "PrimeKG:#5421": {
    id: "PrimeKG:#5421",
    triple: "Metformin —[drug_effect]→ Lactic Acidosis",
    source: "PrimeKG relation: drug_effect (drug_effect_label)",
    confidence: 0.91,
    abstract:
      "Metformin inhibits mitochondrial respiratory-chain complex I, suppressing gluconeogenesis from lactate. In renal impairment, accumulation raises the risk of metformin-associated lactic acidosis (MALA).",
  },
  "PMC:5421": {
    id: "PMC:5421",
    triple: "KDOQI 2020 —[guideline]→ Metformin renal dosing",
    source: "PMC 5421 — KDIGO/KDOQI diabetes management in CKD (mock abstract)",
    confidence: 0.88,
    abstract:
      "Avoid initiating metformin at eGFR < 30 mL/min/1.73m²; for patients on metformin whose eGFR falls below 30, discontinue or reduce dose with close monitoring of renal function and lactate.",
  },
  "PMC:2210": {
    id: "PMC:2210",
    triple: "Lalau et al. —[cohort]→ MALA incidence in CKD",
    source: "PMC 2210 — MALA pharmacovigilance cohort (mock abstract)",
    confidence: 0.82,
    abstract:
      "Metformin-associated lactic acidosis is rare but mortality is high when eGFR < 30. Risk is amplified by acute kidney injury on top of chronic impairment and by iodinated contrast exposure.",
  },
  "PrimeKG:#0710": {
    id: "PrimeKG:#0710",
    triple: "Sitagliptin —[alternative_to]→ Metformin in CKD",
    source: "PrimeKG relation: drug_alternative (inferred, 2-hop)",
    confidence: 0.71,
    abstract:
      "DPP-4 inhibitors are a glucose-lowering alternative with a more favorable renal safety profile; sitagliptin requires dose adjustment by eGFR but does not carry lactic acidosis risk.",
  },
};

const parkinsonCitations: Record<string, CitationRecord> = {
  "PrimeKG:#2201": {
    id: "PrimeKG:#2201",
    triple: "Levodopa —[indication]→ Parkinson's Disease (motor symptoms)",
    source: "PrimeKG relation: indication (drug_indication_label)",
    confidence: 0.95,
    abstract:
      "Levodopa/carbidopa remains the most effective symptomatic therapy for Parkinson's disease motor signs: bradykinesia, rigidity, and resting tremor.",
  },
  "PMC:3301": {
    id: "PMC:3301",
    triple: "Subthalamic DBS —[treats]→ Motor fluctuations in advanced PD",
    source: "PMC 3301 — EARLYSTIM randomized trial report (mock abstract)",
    confidence: 0.9,
    abstract:
      "Bilateral subthalamic-nucleus deep brain stimulation plus medical therapy is superior to medical therapy alone in patients with Parkinson's disease and early motor complications, improving UPDRS-III motor scores and medication-refractory tremor.",
  },
  "PrimeKG:#2204": {
    id: "PrimeKG:#2204",
    triple: "Levodopa —[associated_with]→ Levodopa-induced dyskinesia",
    source: "PrimeKG relation: adverse_event (drug_adverse_event_label)",
    confidence: 0.87,
    abstract:
      "Long-term levodopa exposure is associated with peak-dose dyskinesia and wearing-off fluctuations; amantadine and STN-DBS are recognized management options.",
  },
  "PrimeKG:#2206": {
    id: "PrimeKG:#2206",
    triple: "COMT —[metabolism]→ Levodopa (catechol-O-methyltransferase)",
    source: "PrimeKG relation: enzyme_metabolizes_drug",
    confidence: 0.8,
    abstract:
      "Catechol-O-methyltransferase degrades levodopa peripherally and centrally; entacapone/tolcapone inhibition extends the levodopa plasma half-life and smooths motor response.",
  },
};

const serpinaCitations: Record<string, CitationRecord> = {
  "PrimeKG:#7102": {
    id: "PrimeKG:#7102",
    triple: "SERPINA6 —[encodes]→ Corticosteroid-Binding Globulin (CBG / transcortin)",
    source: "PrimeKG relation: gene_encodes_protein",
    confidence: 0.93,
    abstract:
      "SERPINA6 encodes corticosteroid-binding globulin (transcortin), the principal plasma carrier of glucocorticoids. Roughly 90% of circulating prednisolone is CBG-bound; only the free fraction is pharmacologically active.",
  },
  "PMC:6118": {
    id: "PMC:6118",
    triple: "CBG variants —[modulate]→ Prednisolone free fraction",
    source: "PMC 6118 — pharmacogenetics of glucocorticoid transport (mock abstract)",
    confidence: 0.84,
    abstract:
      "Carriers of SERPINA6 missense variants (e.g., CBG Lyon) show reduced prednisolone-binding capacity, raising the free-drug fraction and increasing exposure at standard doses; therapeutic monitoring is advised in variant carriers.",
  },
  "PrimeKG:#7114": {
    id: "PrimeKG:#7114",
    triple: "Prednisolone —[indication]→ Inflammatory conditions",
    source: "PrimeKG relation: indication",
    confidence: 0.78,
    abstract:
      "Prednisolone is a systemically active glucocorticoid used across inflammatory and autoimmune indications; protein binding to CBG and albumin determines free-drug exposure.",
  },
};

const metforminNodes: GraphNode[] = [
  { id: "metformin", label: "Metformin", type: "drug", trace: 0.86 },
  { id: "ckd", label: "Chronic Kidney Disease", type: "disease", trace: 0.81 },
  { id: "lactic", label: "Lactic Acidosis", type: "disease", trace: 0.78 },
  { id: "t2dm", label: "Type 2 Diabetes", type: "disease", trace: 0.62 },
  { id: "egfr", label: "eGFR < 30", type: "phenotype", trace: 0.74 },
  { id: "oct2", label: "OCT2 / SLC22A2", type: "protein", trace: 0.55 },
  { id: "phenformin", label: "Phenformin", type: "drug", trace: 0.21 },
  { id: "sitagliptin", label: "Sitagliptin", type: "drug", trace: 0.34 },
  { id: "b12", label: "Vitamin B12 deficiency", type: "phenotype", trace: 0.18 },
];

const metforminEdges: GraphEdge[] = [
  { id: "e1", source: "metformin", target: "ckd", relation: "contraindication", weight: 0.94 },
  { id: "e2", source: "metformin", target: "lactic", relation: "drug_effect", weight: 0.91 },
  { id: "e3", source: "metformin", target: "t2dm", relation: "indication", weight: 0.88 },
  { id: "e4", source: "ckd", target: "egfr", relation: "phenotype", weight: 0.83 },
  { id: "e5", source: "metformin", target: "oct2", relation: "carrier", weight: 0.64 },
  { id: "e6", source: "phenformin", target: "lactic", relation: "causes", weight: 0.44 },
  { id: "e7", source: "sitagliptin", target: "ckd", relation: "alternative_to", weight: 0.34 },
  { id: "e8", source: "metformin", target: "b12", relation: "associated_with", weight: 0.25 },
];

const parkinsonNodes: GraphNode[] = [
  { id: "pd", label: "Parkinson's Disease", type: "disease", trace: 0.84 },
  { id: "levodopa", label: "Levodopa", type: "drug", trace: 0.86 },
  { id: "dbs", label: "Subthalamic DBS", type: "phenotype", trace: 0.79 },
  { id: "stn", label: "Subthalamic Nucleus", type: "phenotype", trace: 0.66 },
  { id: "tremor", label: "Motor Tremor", type: "phenotype", trace: 0.72 },
  { id: "dyskinesia", label: "Dyskinesia", type: "adverse_effect", trace: 0.58 },
  { id: "comt", label: "COMT", type: "protein", trace: 0.51 },
  { id: "d2r", label: "D2 Receptor", type: "protein", trace: 0.38 },
];

const parkinsonEdges: GraphEdge[] = [
  { id: "e1", source: "levodopa", target: "pd", relation: "indication", weight: 0.95 },
  { id: "e2", source: "dbs", target: "tremor", relation: "treats", weight: 0.88 },
  { id: "e3", source: "dbs", target: "stn", relation: "targets", weight: 0.82 },
  { id: "e4", source: "levodopa", target: "dyskinesia", relation: "associated_with", weight: 0.79 },
  { id: "e5", source: "comt", target: "levodopa", relation: "metabolism", weight: 0.68 },
  { id: "e6", source: "pd", target: "tremor", relation: "phenotype", weight: 0.74 },
  { id: "e7", source: "levodopa", target: "d2r", relation: "agonist", weight: 0.4 },
];

const serpinaNodes: GraphNode[] = [
  { id: "serpina6", label: "SERPINA6", type: "gene", trace: 0.85 },
  { id: "cbg", label: "CBG / Transcortin", type: "protein", trace: 0.82 },
  { id: "prednisolone", label: "Prednisolone", type: "drug", trace: 0.8 },
  { id: "albumin", label: "Albumin", type: "protein", trace: 0.42 },
  { id: "inflammation", label: "Inflammation", type: "disease", trace: 0.55 },
  { id: "dexamethasone", label: "Dexamethasone", type: "drug", trace: 0.28 },
];

const serpinaEdges: GraphEdge[] = [
  { id: "e1", source: "serpina6", target: "cbg", relation: "encodes", weight: 0.93 },
  { id: "e2", source: "prednisolone", target: "cbg", relation: "binds", weight: 0.9 },
  { id: "e3", source: "prednisolone", target: "albumin", relation: "binds", weight: 0.46 },
  { id: "e4", source: "prednisolone", target: "inflammation", relation: "indication", weight: 0.7 },
  { id: "e5", source: "dexamethasone", target: "cbg", relation: "competes", weight: 0.31 },
];

export const CASES: ClinicalCase[] = [
  {
    id: "metformin-ckd",
    label: "Metformin + CKD",
    query: "Can Metformin be given to patients with severe renal failure given lactic acidosis risk?",
    keywords: ["metformin", "renal", "kidney", "lactic", "acidosis", "ckd", "egfr"],
    entities: [
      { name: "Metformin", type: "drug", nodeId: "PrimeKG:DrugBank:DB00331" },
      { name: "Chronic Kidney Disease", type: "disease", nodeId: "PrimeKG:MONDO:0005300" },
      { name: "Lactic Acidosis", type: "disease", nodeId: "PrimeKG:MONDO:0009553" },
      { name: "eGFR < 30", type: "phenotype" },
      { name: "OCT2", type: "gene", nodeId: "PrimeKG:NCBIGene:65280" },
    ],
    candidates: [
      { text: "Metformin — contraindication → CKD (eGFR < 30 mL/min/1.73m²)", score: 0.86, source: "PrimeKG", redundancy: 0.05 },
      { text: "Metformin — drug_effect → Lactic Acidosis (MALA risk)", score: 0.83, source: "PrimeKG", redundancy: 0.11 },
      { text: "Metformin — indication → Type 2 Diabetes Mellitus", score: 0.79, source: "PrimeKG", redundancy: 0.18 },
      { text: "KDOQI 2020: avoid metformin at eGFR < 30; monitor if 30–45", score: 0.77, source: "PMC", redundancy: 0.24 },
      { text: "Metformin — carrier → OCT2 (SLC22A2) renal tubular uptake", score: 0.64, source: "PrimeKG", redundancy: 0.31 },
      { text: "Metformin — transporter → SLC22A1 (OCT1) hepatic uptake", score: 0.61, source: "PrimeKG", redundancy: 0.38 },
      { text: "Metformin accumulation raises lactate in severe renal impairment", score: 0.59, source: "PMC", redundancy: 0.42 },
      { text: "Metformin — associated_with → Vitamin B12 deficiency", score: 0.55, source: "PrimeKG", redundancy: 0.47 },
      { text: "Sulfonylurea hypoglycemia risk in CKD (alternative agent class)", score: 0.51, source: "PMC", redundancy: 0.52 },
      { text: "DPP-4 inhibitors renally dosed alternatives in CKD", score: 0.48, source: "PMC", redundancy: 0.55 },
      { text: "Phenformin — causes → Lactic Acidosis (withdrawn biguanide)", score: 0.44, source: "PrimeKG", redundancy: 0.71 },
      { text: "Metformin starting dose 500 mg BID with meals", score: 0.41, source: "PMC", redundancy: 0.58 },
      { text: "SGLT2 inhibitors slow CKD progression (dapagliflozin)", score: 0.38, source: "PMC", redundancy: 0.49 },
      { text: "Ibuprofen — associated_with → Acute kidney injury", score: 0.33, source: "PrimeKG", redundancy: 0.2 },
      { text: "Vancomycin — associated_with → Nephrotoxicity", score: 0.27, source: "PrimeKG", redundancy: 0.17 },
      { text: "CRABP2 — carrier → retinoid transport", score: 0.21, source: "PrimeKG", redundancy: 0.08 },
      { text: "Mycophenolate — metabolism → CYP3A5", score: 0.19, source: "PrimeKG", redundancy: 0.06 },
      { text: "Sunitinib — metabolism → CYP3A4", score: 0.17, source: "PrimeKG", redundancy: 0.05 },
      { text: "Warfarin — metabolism → CYP2C9", score: 0.15, source: "PrimeKG", redundancy: 0.04 },
      { text: "Atorvastatin — transporter → OATP1B1", score: 0.13, source: "PrimeKG", redundancy: 0.03 },
    ],
    citations: metforminCitations,
    subgraph: {
      caseId: "metformin-ckd",
      centerId: "metformin",
      hops: 2,
      nodes: metforminNodes,
      edges: metforminEdges,
    },
    arena: {
      baseline: [
        { kind: "text", value: "Metformin is generally safe; many patients with mild kidney disease can continue it. A common starting dose is 500 mg twice daily and can be titrated up in renal impairment with care." },
        { kind: "text", value: "[!] Ungrounded dosing risk: suggestion to titrate in renal impairment without eGFR thresholds and without any cited source. No mention of the eGFR < 30 contraindication — a documented hallucination pattern for this medication class." },
      ],
      vanillaRag: [
        { kind: "text", value: "Retrieved context (unconditional top-K): fragments about retinoid carriers, CYP3A5 metabolism, and unrelated drug interactions crowd out the two clinically decisive triples." },
        { kind: "text", value: "Context dominated by distractors: CRABP2 retinoid carrier, mycophenolate/CYP3A5, sunitinib/CYP3A4. Fragmented sentences with no clinical reasoning chain; dosing guidance missing because relevant chunks lost the rank race." },
      ],
      triage: [
        { kind: "text", value: "Metformin is contraindicated at eGFR < 30 mL/min/1.73m² due to accumulation and metformin-associated lactic acidosis (MALA) risk " },
        { kind: "citation", id: "PrimeKG:#8921" },
        { kind: "citation", id: "PrimeKG:#5421" },
        { kind: "text", value: ". If therapy must continue between eGFR 30–45, reduce dose and monitor renal function and lactate closely " },
        { kind: "citation", id: "PMC:5421" },
        { kind: "text", value: ". Alternative glucose-lowering with a more favorable renal profile: DPP-4 inhibitors with eGFR-adjusted dosing " },
        { kind: "citation", id: "PrimeKG:#0710" },
        { kind: "text", value: "." },
      ],
      warnings: [
        "Baseline answer omits the eGFR < 30 contraindication — high-severity omission.",
        "Vanilla RAG context window: 62% of tokens spent on clinically irrelevant distractors.",
      ],
    },
  },
  {
    id: "parkinson-dbs",
    label: "PD: Levodopa vs DBS",
    query: "Parkinson's disease motor tremor: levodopa vs subthalamic nucleus DBS?",
    keywords: ["parkinson", "levodopa", "dbs", "subthalamic", "tremor", "motor"],
    entities: [
      { name: "Parkinson's Disease", type: "disease", nodeId: "PrimeKG:MONDO:0005180" },
      { name: "Levodopa", type: "drug", nodeId: "PrimeKG:DrugBank:DB01235" },
      { name: "Subthalamic DBS", type: "phenotype" },
      { name: "Motor Tremor", type: "phenotype" },
      { name: "Dyskinesia", type: "adverse_effect" },
      { name: "COMT", type: "gene", nodeId: "PrimeKG:NCBIGene:1312" },
    ],
    candidates: [
      { text: "Levodopa — indication → Parkinson's Disease motor symptoms", score: 0.84, source: "PrimeKG", redundancy: 0.06 },
      { text: "STN-DBS improves motor fluctuations and refractory tremor", score: 0.81, source: "PMC", redundancy: 0.13 },
      { text: "EARLYSTIM: DBS + medical therapy superior for early motor complications", score: 0.74, source: "PMC", redundancy: 0.21 },
      { text: "Levodopa — associated_with → Levodopa-induced dyskinesia", score: 0.72, source: "PrimeKG", redundancy: 0.26 },
      { text: "Carbidopa blocks peripheral L-DOPA decarboxylation", score: 0.68, source: "PMC", redundancy: 0.33 },
      { text: "COMT — metabolism → Levodopa (entacapone extends half-life)", score: 0.66, source: "PrimeKG", redundancy: 0.36 },
      { text: "Amantadine reduces levodopa-induced dyskinesia severity", score: 0.52, source: "PMC", redundancy: 0.44 },
      { text: "MAO-B inhibitors (rasagiline) as adjunctive therapy", score: 0.49, source: "PMC", redundancy: 0.48 },
      { text: "Tremor-dominant PD: DBS candidacy evaluation criteria", score: 0.47, source: "PMC", redundancy: 0.51 },
      { text: "Dopamine — agonist → D2 receptor signaling", score: 0.42, source: "PrimeKG", redundancy: 0.39 },
      { text: "Apomorphine rescue for off episodes", score: 0.4, source: "PMC", redundancy: 0.43 },
      { text: "Fetal dopaminergic cell transplantation trial history", score: 0.24, source: "PMC", redundancy: 0.22 },
      { text: "Coenzyme Q10 neuroprotection (refuted)", score: 0.19, source: "PMC", redundancy: 0.14 },
      { text: "Rasagiline — metabolism → CYP1A2", score: 0.17, source: "PrimeKG", redundancy: 0.11 },
      { text: "Selegiline cheese effect (tyramine)", score: 0.13, source: "PMC", redundancy: 0.07 },
    ],
    citations: parkinsonCitations,
    subgraph: {
      caseId: "parkinson-dbs",
      centerId: "pd",
      hops: 2,
      nodes: parkinsonNodes,
      edges: parkinsonEdges,
    },
    arena: {
      baseline: [
        { kind: "text", value: "Levodopa is the standard treatment for Parkinson's tremor. Deep brain stimulation cures Parkinson's disease in advanced stages and can be applied early instead of medication." },
        { kind: "text", value: "[!] Hallucination risk: 'DBS cures PD' and 'DBS instead of medication early' are ungrounded claims. DBS is symptomatic therapy for medication-refractory motor complications; no source cited for any claim." },
      ],
      vanillaRag: [
        { kind: "text", value: "Retrieved context (unconditional top-K): a mixed bag of UPDRS score tables, transplantation trial history, and CoQ10 neuroprotection fragments." },
        { kind: "text", value: "Distractors dominate: fetal cell transplantation history and refuted CoQ10 neuroprotection consume the context budget; the levodopa–dyskinesia relation never surfaces because chunking broke the trial abstract mid-sentence." },
      ],
      triage: [
        { kind: "text", value: "Levodopa remains first-line for PD motor signs including tremor " },
        { kind: "citation", id: "PrimeKG:#2201" },
        { kind: "text", value: ". For medication-refractory motor fluctuations and dyskinesia, bilateral subthalamic DBS plus medical therapy outperforms medication alone " },
        { kind: "citation", id: "PMC:3301" },
        { kind: "text", value: ". Long-term levodopa carries peak-dose dyskinesia risk — amantadine or COMT inhibition can smooth response " },
        { kind: "citation", id: "PrimeKG:#2204" },
        { kind: "citation", id: "PrimeKG:#2206" },
        { kind: "text", value: "." },
      ],
      warnings: [
        "Baseline asserts DBS is curative — contradicted by EARLYSTIM trial framing.",
        "Vanilla RAG retrieved refuted neuroprotection claims (CoQ10) into the context window.",
      ],
    },
  },
  {
    id: "serpina6-prednisolone",
    label: "SERPINA6 / Prednisolone",
    query: "SERPINA6 gene variant impact on prednisolone pharmacokinetics?",
    keywords: ["serpina6", "prednisolone", "pharmacokinetics", "cbg", "transcortin", "corticosteroid"],
    entities: [
      { name: "SERPINA6", type: "gene", nodeId: "PrimeKG:NCBIGene:506" },
      { name: "Prednisolone", type: "drug", nodeId: "PrimeKG:DrugBank:DB00924" },
      { name: "Corticosteroid-Binding Globulin", type: "protein" },
      { name: "Glucocorticoid Resistance", type: "phenotype" },
    ],
    candidates: [
      { text: "SERPINA6 encodes corticosteroid-binding globulin (CBG/transcortin)", score: 0.85, source: "PrimeKG", redundancy: 0.05 },
      { text: "Prednisolone — binds → CBG (~90% of circulating drug)", score: 0.8, source: "PrimeKG", redundancy: 0.12 },
      { text: "CBG missense variants raise free prednisolone fraction", score: 0.76, source: "PMC", redundancy: 0.19 },
      { text: "CBG Lyon variant: reduced binding capacity, higher free fraction", score: 0.62, source: "PMC", redundancy: 0.28 },
      { text: "Albumin binds the remaining prednisolone fraction at high doses", score: 0.55, source: "PMC", redundancy: 0.35 },
      { text: "Prednisolone — indication → inflammatory conditions", score: 0.5, source: "PrimeKG", redundancy: 0.41 },
      { text: "Dexamethasone — competes → CBG binding sites", score: 0.43, source: "PrimeKG", redundancy: 0.46 },
      { text: "SERPINA1 encodes alpha-1-antitrypsin (paralogous locus)", score: 0.22, source: "PrimeKG", redundancy: 0.14 },
      { text: "Cortisol — binds → mineralocorticoid receptor", score: 0.18, source: "PrimeKG", redundancy: 0.09 },
      { text: "Metyrapone inhibits 11β-hydroxylase", score: 0.12, source: "PrimeKG", redundancy: 0.05 },
    ],
    citations: serpinaCitations,
    subgraph: {
      caseId: "serpina6-prednisolone",
      centerId: "serpina6",
      hops: 2,
      nodes: serpinaNodes,
      edges: serpinaEdges,
    },
    arena: {
      baseline: [
        { kind: "text", value: "SERPINA6 is involved in steroid biology. Prednisolone is a corticosteroid that is metabolized by the liver; genetic variants generally do not change dosing." },
        { kind: "text", value: "[!] Conflation risk: answer brushes past SERPINA6's carrier role and asserts variants do not change exposure — contradicting pharmacogenetic evidence. No binding-fraction quantification, no source." },
      ],
      vanillaRag: [
        { kind: "text", value: "Retrieved context (unconditional top-K): fragments on SERPINA1 (alpha-1-antitrypsin), mineralocorticoid receptors, and metyrapone enzyme chemistry." },
        { kind: "text", value: "Wrong-gene distractor: SERPINA1 fragments are lexically similar to SERPINA6 and outrank the CBG-binding abstracts; the context window carries zero usable pharmacokinetic evidence." },
      ],
      triage: [
        { kind: "text", value: "SERPINA6 encodes corticosteroid-binding globulin (CBG), the principal plasma carrier of prednisolone " },
        { kind: "citation", id: "PrimeKG:#7102" },
        { kind: "text", value: ". Missense variants such as CBG Lyon reduce binding capacity and raise the free (active) prednisolone fraction, increasing exposure at standard doses — therapeutic drug monitoring is advised in variant carriers " },
        { kind: "citation", id: "PMC:6118" },
        { kind: "text", value: "." },
      ],
      warnings: [
        "Baseline dismisses pharmacogenetic effect — contradicted by CBG binding literature.",
        "Vanilla RAG retrieved SERPINA1 (paralog) instead of SERPINA6 evidence.",
      ],
    },
  },
];

/** Flat dictionary used for live SapBERT-style chip extraction in standalone mode. */
export const ENTITY_DICTIONARY: { name: string; type: EntityType; caseId: string }[] =
  CASES.flatMap((c) =>
    c.entities.map((e) => ({ name: e.name, type: e.type, caseId: c.id }))
  );

export function resolveCase(query: string): ClinicalCase {
  const q = query.toLowerCase();
  let best = CASES[0];
  let bestScore = 0;
  for (const c of CASES) {
    const score = c.keywords.reduce(
      (acc, k) => acc + (q.includes(k) ? 1 : 0),
      0
    );
    if (score > bestScore) {
      bestScore = score;
      best = c;
    }
  }
  return best;
}

export function extractEntities(query: string): { name: string; type: EntityType }[] {
  const q = query.toLowerCase();
  const seen = new Set<string>();
  const out: { name: string; type: EntityType }[] = [];
  for (const e of ENTITY_DICTIONARY) {
    const key = e.name.toLowerCase();
    if (q.includes(key) && !seen.has(e.name)) {
      seen.add(e.name);
      out.push({ name: e.name, type: e.type });
    }
  }
  return out;
}

export const BENCHMARK_MOCK: BenchmarkResponse = {
  radar: [
    { dimension: "Diagnostic Accuracy", triage: 88, vanilla: 71, baseline: 62 },
    { dimension: "Evidence Faithfulness", triage: 92, vanilla: 64, baseline: 51 },
    { dimension: "Contradiction Avoidance", triage: 85, vanilla: 58, baseline: 49 },
    { dimension: "Hallucination Suppression", triage: 89, vanilla: 55, baseline: 47 },
    { dimension: "Completeness", triage: 84, vanilla: 72, baseline: 66 },
  ],
  winRate: [
    { criterion: "Comprehensiveness", rate: 61.9 },
    { criterion: "Empowerment", rate: 66.7 },
    { criterion: "Accuracy", rate: 57.1 },
    { criterion: "Relevance", rate: 52.4 },
    { criterion: "Coherence", rate: 71.4 },
    { criterion: "Clarity", rate: 66.7 },
    { criterion: "Logical", rate: 61.9 },
    { criterion: "Flexibility", rate: 47.6 },
  ],
  mcnemar: {
    statistic: 18.4,
    pValue: 0.000018,
    significant: true,
    n: 50,
  },
  abstentionRate: 18.4,
  nQueries: 50,
};
