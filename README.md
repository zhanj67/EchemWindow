# MatterChat-ECW: Structure-Aware Electrochemical Window Prediction & Atom-Scale Redox Attribution

## Project Prompt & Objective

> **Prompt:** Design and implement a structure-aware multi-modal pipeline that adapts **MatterChat** (frozen CHGNet encoder + trainable BLIP-2 Bridge Model + frozen Mistral 7B) to the **`ecw_mp_16k.csv`** Li-containing solid-state electrolyte (SSE) dataset. Your workflow must go beyond composition-only stoichiometry screening by:
> 1. Resolving polymorph and data-leakage pitfalls inherent to Grand Potential Phase Diagram (GPPD) labels.
> 2. Fine-tuning the Bridge Model on multi-task ECW regression (`Left End (V)`, `Right End (V)`, `Window Length (V)`) and wide-ECW classification (`label`, $\ge 3\text{ V}$) alongside MatterChat's native stability and electronic tasks.
> 3. Extracting **atom-level importance scores** via gradient-weighted cross-attention to identify redox-limiting cationic and anionic sites in 3D crystal structures.

---

## 1. Dataset Audit (`ecw_mp_16k.csv`)

Before modeling, the pipeline must account for four empirical characteristics observed directly in `ecw_mp_16k.csv`:

| Metric / Column | Observed Value / Range | Modeling Implication |
| :--- | :--- | :--- |
| **Total Rows** | 16,001 entries (0 nulls) | Clean tabular labels, ready for `MP ID` structure joins |
| **Unique `MP ID`** | 15,981 (20 duplicates) | Deduplicate by `MP ID` prior to splitting |
| **Unique `Formula`** | 5,939 (~10,062 polymorph entries) | **Critical:** Must use **composition-grouped splits** (GroupKFold on `Formula`) to prevent polymorph leakage across train/val/test |
| **`Band Gap (eV)`** | $0.00 - 8.08\text{ eV}$ (mean $1.19\text{ eV}$) | Overlaps with MatterChat's native bandgap task; enables multi-task consistency checks |
| **`energy_above_hull (eV/atom)`** | $0.00 - 7.61\text{ eV/atom}$ (mean $0.12\text{ eV/atom}$) | Overlaps with MatterChat's native $E_{\text{hull}}$ task; key filter ($< 0.03\text{ eV/atom}$) |
| **`Left End (V)`** | $0.01 - 5.45\text{ V}$ (mean $2.47 \pm 0.90\text{ V}$) | Reduction limit vs. $\text{Li}/\text{Li}^+$ |
| **`Right End (V)`** | $0.01 - 7.23\text{ V}$ (mean $3.26 \pm 0.84\text{ V}$) | Oxidation limit vs. $\text{Li}/\text{Li}^+$ |
| **`Window Length (V)`** | $0.00 - 6.01\text{ V}$ (mean $0.79 \pm 0.70\text{ V}$) | $\text{Right End} - \text{Left End}$ |
| **`label` (ECW $\ge 3\text{ V}$)** | 130 positives (0.81% Class 1) | Severe class imbalance; evaluate with **ROC-AUC / PR-AUC**, not raw accuracy alone |

### The "GPPD Trap" to Design Around
Because GPPD-derived ECW labels are strictly thermodynamic, polymorphs sharing the same `Formula` (e.g., `mp-1026818` and `mp-1026831`, both `SrLiMg14`) carry identical `Left End (V)` and `Right End (V)` values despite differing in 3D structure and `energy_above_hull`.
* **Never split randomly by `MP ID`**—identical formulas will leak across splits and artificially inflate $R^2$.
* **Use 3D structure where it adds real signal:** polymorph $E_{\text{hull}}$ differentiation, unified multi-objective screening, and atom-level redox-site attribution.

---

## 2. Proposed End-to-End Workflow

### Stage 1: Structure Alignment & Leakage-Free Splitting
1. **Deduplicate** `ecw_mp_16k.csv` on `MP ID` (retain 15,981 unique structures).
2. **Fetch 3D Crystal Structures** for each `MP ID` from the Materials Project API (`mp-api`) or relaxed **MPtrj** snapshots so inputs match MatterChat's CHGNet pretraining distribution.
3. **Group-Split by `Formula` (80% / 10% / 10%)** using stratified group splitting (stratified on `label` and anion class: oxide, sulfide, halide, mixed-anion) so no chemical composition appears in both training and evaluation.
4. **Generate Multi-Modal QA Pairs** following MatterChat's Phase 2 instruction templates for:
   * `Left End (V)` (reduction limit)
   * `Right End (V)` (oxidation limit)
   * `Window Length (V)` and binary `label` (ECW $\ge 3\text{ V}$)
   * Auxiliary co-tasks already present in the CSV: `Band Gap (eV)` and `energy_above_hull (eV/atom)`

### Stage 2: Parameter-Efficient Multi-Task Fine-Tuning (MatterChat Phase 2)
1. **Freeze** the CHGNet uMLIP graph encoder and Mistral 7B LLM backbone.
2. **Fine-tune only the BLIP-2 Bridge Model** (32 trainable queries + alternating self/cross-attention + linear projection) on the ECW + auxiliary QA pairs using supervised next-token cross-entropy loss.
3. **Handle Class Imbalance (`label = 1` at 0.81%):**
   * Do **not** downsample Class 0 (prior benchmarks show downsampling degrades generalization on unseen compounds).
   * Instead, use task-balanced sampling or focal/class-weighted loss on the binary classification templates, and monitor **ROC-AUC** and **PR-AUC**.
4. **Inference Stabilization (Multi-Modal RAG):**
   * Index the training set's flattened `(32, 4096)` Bridge embeddings.
   * At inference, retrieve the top-2 nearest neighbors by $L_2$ distance and aggregate numerical ECW predictions (mean) and class predictions (majority vote).

### Stage 3: Atom-Scale Redox Attribution (Core Scientific Contribution)
Extract per-atom importance scores $s_i$ directly from the Bridge Model's cross-attention weights $A \in \mathbb{R}^{H \times Q \times N_{\text{atoms}}}$:

1. **Gradient-Weighted Cross-Attention:**
   * Run separate forward passes for `Left End (V)` (reduction) and `Right End (V)` (oxidation) prompts.
   * Backpropagate the target numeric token logit $y$ to the final cross-attention weights $A$ (weights remain frozen; retain gradients on $A$).
   * Compute per-atom attribution:
     $$s_i = \sum_{h, q} \text{ReLU}\!\left( A_{h, q, i} \odot \frac{\partial y}{\partial A_{h, q, i}} \right)$$
2. **Integrated Gradients (Cross-Check):**
   * Compute Integrated Gradients directly on CHGNet's per-atom embeddings $H_i$ for a subset of top SSE candidates to verify attribution stability across non-linear layers.
3. **Physical Validation Suite:**
   * **Wyckoff Symmetry Test:** Crystallographically equivalent sites must yield identical $s_i$ scores.
   * **Redox Contrast Test:** Confirm that `Left End (V)` attribution concentrates on reducible cations (e.g., transition metals) while `Right End (V)` attribution concentrates on the anion framework (e.g., $\text{O}^{2-}, \text{S}^{2-}, \text{Cl}^-, \text{F}^-$).
   * **DFT PDOS Alignment:** Compare top-attributed atomic sites against Projected Density of States (PDOS) near the VBM and CBM for representative halides, oxides, and sulfides.

### Stage 4: Unified Screening & Mechanistic Reporting
1. **Single-Pass Multi-Objective Filter:** Screen unseen structures (e.g., OQMD / GNoME) in one model pass using `energy_above_hull < 0.03 eV/atom`, `Band Gap > 3.0 eV`, and `Window Length >= 3.0 V`.
2. **Export 3D Visuals:** Write per-atom $s_i$ scores into `.extxyz` / `.cif` files for visualization in VESTA or OVITO.
3. **LLM Synthesis & Mechanistic Reasoning:** Prompt the Mistral 7B head on top candidates to propose solid-state synthesis routes and explain how local coordination environments govern redox stability.

---

## 3. Recommended Baselines & Evaluation Metrics

| Model / Variant | Input | Primary Metrics | Purpose |
| :--- | :--- | :--- | :--- |
| **Roost (Composition-Only)** | `Formula` | $R^2$, MAE, RMSE (Left/Right); ROC-AUC (`label`) | Baseline from the original *J. Mater. Chem. A* study |
| **CHGNet + MLP Head** | 3D Graph | $R^2$, MAE, RMSE; ROC-AUC | Isolates the value of 3D structure without the LLM |
| **MatterChat-ECW (Bridge FT)** | 3D Graph + Prompt | $R^2$, MAE, RMSE; ROC-AUC; PR-AUC | Evaluates unified multi-task alignment |
| **MatterChat-ECW + RAG** | 3D Graph + Prompt + $L_2$ Retrieval | $R^2$, MAE, RMSE; ROC-AUC; PR-AUC | Tests numerical stabilization via embedding retrieval |

---

## 4. Suggested Repository Structure

```text
matterchat-ecw/
├── data/
│   ├── ecw_mp_16k.csv              # Raw ECW dataset (16,001 rows)
│   ├── structures/                 # Relaxed CIF / pymatgen JSON structures by MP ID
│   └── splits/                     # Formula-grouped train/val/test JSONs
├── src/
│   ├── data_prep.py                # Deduplication, MP structure fetch, QA template builder
│   ├── train_bridge.py             # Phase 2 Bridge Model fine-tuning (CHGNet & LLM frozen)
│   ├── rag_inference.py            # L2-similarity multi-modal RAG evaluation
│   ├── atom_attribution.py         # PyTorch hooks for grad-weighted cross-attention & IG
│   └── validate_physics.py         # Wyckoff symmetry & redox-contrast sanity checks
├── notebooks/
│   └── visualize_redox_sites.ipynb # Export atom scores to extxyz/CIF for VESTA/OVITO
└── README.md
```
