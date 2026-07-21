# 数据集防质疑调研（31篇论文实证，2026-06-27）

来源: references/{selection,multimodal}/ 共31篇，多智能体抽取各自用的数据集后综合。

## 各模态公认数据集

### language
- 事实标准: FineWeb-Edu (the single most defensible English-language selection pool: it is the headline NeurIPS 2024 FineWeb artifact AND is reused as the language source in SmolLM2; reviewers universally accept it).
- 标准评测: Held-out perplexity plus the standard LM-eval downstream cluster used across DCLM/FineWeb/SmolLM2/RegMix/QuRating: MMLU, ARC-E/ARC-C, HellaSwag, PIQA, OpenBookQA, WinoGrande, CommonsenseQA, SciQ. For a perplexity-only pipeline, held-out FineWeb-Edu / C4 perplexity is the accepted metric.
- 文献用过的:
  - FineWeb-Edu / FineWeb (FineWeb, NeurIPS 2024; reused directly as a selection source in SmolLM2, arXiv 2025; FineWeb-Edu classifier reused as a filter signal in QuaDMix and RefinedWeb-based pools) -- highest recognition, top-venue
  - C4 (DsDm, arXiv 2024; Ask-LLM, arXiv 2024; DoGE, ICLR 2024; SemDeDup auxiliary, ICLR 2023) -- the classic controlled selection pool
  - SlimPajama (DoGE, ICLR 2024; QuRating, ICML/PMLR 2024; DEITA, ICLR 2024) -- common multi-domain pool
  - RedPajama / RedPajama-V2 (QuRating, ICML 2024; PerplexityCorrelations, arXiv 2024) -- web-scale selection pool
  - The Pile (DoReMi, ICLR 2023; RegMix, arXiv 2024) -- domain-reweighting benchmark pool
  - CommonCrawl / RefinedWeb / DCLM (D4, arXiv 2023; DCLM, 2024/2025; FineWeb, NeurIPS 2024; QuaDMix, 2024; SmolLM2) -- raw upstream pool

### math
- 事实标准: OpenWebMath for pretraining-style math web text; FineMath-4plus is an equally defensible, more modern choice because it is the direct math sibling of FineWeb-Edu and is used by SmolLM2 with explicit quality tiers (the '4plus' tier is a real, recognized split).
- 标准评测: GSM8K and MATH (universal across OpenWebMath, DeepSeekMath, MathPile, SmolLM2), plus MMLU-STEM. For a perplexity pipeline, held-out OpenWebMath/FineMath perplexity.
- 文献用过的:
  - OpenWebMath (OpenWebMath, arXiv 2023; reused as a pool/source in DeepSeekMath, arXiv 2024, and SmolLM2) -- most-cited math pretraining pool
  - FineMath / FineMath-4plus (SmolLM2, arXiv 2025 -- the FineWeb-team math corpus, the math counterpart to FineWeb-Edu) -- newest recognized standard
  - MathPile (NeurIPS 2024) -- billion-token curated math corpus
  - MathInstruct (S2L, NeurIPS 2024) -- math SFT selection pool (instruction, not pretraining)

### code
- 事实标准: The Stack (v1/v2) or its quality-filtered subset Stack-Edu / StarCoderData. These are the only code pools that appear in this literature; Stack-Edu is the most aligned with a quality-selection framing.
- 标准评测: HumanEval and MBPP (literally every code paper here: The Stack, phi-1, StarCoder2). For a perplexity pipeline, held-out The Stack / Stack-Edu Python perplexity.
- 文献用过的:
  - The Stack / The Stack v2 (The Stack, arXiv 2022; StarCoder2 + Stack v2, arXiv 2024; phi-1 'Textbooks Are All You Need', arXiv 2023) -- de-facto standard permissively-licensed code pool
  - StarCoderData / Stack-Edu (SmolLM2, arXiv 2025) -- the educational/quality-filtered code subset, the code counterpart to FineWeb-Edu
  - GitHub raw (upstream source for The Stack, DoGE's GitHub domain, StarCoder2)

### vision-language
- 事实标准: DataComp (for pretraining-scale selection) or LLaVA-665K (for instruction-tuning selection). Critically, EVERY vision-language data-selection paper in this catalog operates on image-text PAIRS and evaluates with CLIP zero-shot / VQA / retrieval. There is NO precedent in this literature for serializing captions to text and reporting text perplexity.
- 标准评测: ImageNet zero-shot + COCO/Flickr retrieval (DataComp/SemDeDup/T-MARS/MetaCLIP); or VQAv2/GQA/TextVQA/POPE/MME/MMBench (instruction-tuning papers). None of these is reproducible by a text-only perplexity LM.
- 文献用过的:
  - DataComp (DataComp, arXiv 2023; T-MARS, ICLR 2024) -- de-facto multimodal selection benchmark, image-text PAIRS evaluated by CLIP zero-shot
  - LAION / LAION-2B/440M (SemDeDup, ICLR 2023; T-MARS, ICLR 2024) -- web image-text pairs
  - CommonCrawl-derived image-text (MetaCLIP, ICLR 2024) -- 400M pairs
  - LLaVA-665K / Vision-Flan / Cambrian-7M (Self-Filter; TIVE; COINCIDE, CVPR 2024; ICONS) -- visual-instruction-tuning selection pools
  - COCO captions (appears ONLY as an eval/retrieval benchmark or as a source inside LLaVA/TIVE, NEVER as a standalone text-only language pool)

### table
- 事实标准: None exists in this literature. There is no recognized table data-selection benchmark in the surveyed data-selection literature.
- 标准评测: None in this literature.
- 文献用过的:
  - NONE. No paper in the 31-paper catalog uses any tabular dataset as a data-selection pool or benchmark. Adult/Census Income (mstz/adult) appears in zero papers.

### timeseries
- 事实标准: None exists in this literature. There is no recognized time-series data-selection benchmark among these 31 papers.
- 标准评测: None in this literature.
- 文献用过的:
  - NONE. No paper in the catalog uses any time-series dataset as a selection pool or benchmark. Chronos kernel_synth appears in zero papers.

## 当前池审计

### language — DEFENSIBLE
- 现用: FineWeb-Edu
- 建议: Keep FineWeb-Edu. (Optional robustness: add a second classic pool such as C4 or SlimPajama for a cross-pool generalization check, since reviewers like seeing the method transfer across pools.)
- 原因: FineWeb-Edu is the strongest possible choice: it is the headline artifact of FineWeb (NeurIPS 2024) and is reused as the language source by SmolLM2. It is unimpeachable for a perplexity-based language pool.

### math — DEFENSIBLE
- 现用: FineMath-4plus
- 建议: Keep FineMath-4plus. If a reviewer wants a more-cited classic, OpenWebMath is the fallback, but FineMath is the modern FineWeb-team standard used by SmolLM2 and the '4plus' quality tier directly fits a quality-selection narrative.
- 原因: FineMath is recognized via SmolLM2 (arXiv 2025) and is the math sibling of FineWeb-Edu from the same team. The 4plus tier is a real, named quality split. OpenWebMath is the only equally/more-cited alternative, so the choice is safe.

### code — WEAK
- 现用: codeparrot-clean-valid (fallback; Python-Edu was gated)
- 建议: Swap to Stack-Edu (SmolLM2) or StarCoderData / The Stack v2 Python subset. If gating/access is the blocker, use bigcode/the-stack-smol or the public StarCoderData Python split rather than codeparrot.
- 原因: codeparrot-clean appears in ZERO of the 31 papers. The recognized code pools are The Stack / Stack v2 / Stack-Edu / StarCoderData (The Stack 2022, StarCoder2 2024, phi-1 2023, SmolLM2 2025). codeparrot is an old GitHub-Python dump with no standing in the data-selection literature; a reviewer can call it non-standard. Python-Edu (the original plan) is exactly right because it is the code analogue of FineWeb-Edu/FineMath; use any public Stack-Edu/StarCoderData Python split to recover that defensibility.

### vision-language (image) — INDEFENSIBLE
- 现用: COCO captions TEXT only (serialized as text for a text LM)
- 建议: Either (a) DROP the vision-language modality from the perplexity pipeline, or (b) if a multimodal claim is required, run a genuine image-text selection pool (DataComp or LLaVA-665K) under its real CLIP/VQA protocol in a separate track. Do NOT present COCO-caption-text perplexity as 'vision-language data selection'.
- 原因: Every vision-language data-selection paper in this catalog (SemDeDup, DataComp, MetaCLIP, T-MARS, Self-Filter, TIVE, COINCIDE/CVPR2024, ICONS) operates on image-text PAIRS and evaluates with CLIP zero-shot or VQA. Serializing COCO captions to text strips the visual modality entirely, so it is just generic English text, not vision-language. COCO captions never appear as a text-only selection pool in this literature; a reviewer will say the visual modality is absent and the label is misleading.

### table — INDEFENSIBLE
- 现用: mstz/adult (Adult/Census Income, serialized rows)
- 建议: DROP. No recognized data-selection benchmark exists for tables in this literature. If a tabular claim is essential, it belongs to a separate tabular line of work (e.g., TabPFN/tabular-FM literature), not this data-selection paper, and cannot be defended with this 31-paper baseline set.
- 原因: Zero of the 31 papers use any tabular pool or benchmark, and Adult/Census appears nowhere. Serializing Adult rows to text and reporting perplexity has no precedent or comparison point in data-selection work; a reviewer has no anchor and will reject it as ad hoc. Adult is also a toy fairness dataset, which weakens it further.

### timeseries — INDEFENSIBLE
- 现用: chronos kernel_synth (synthetic numeric series serialized)
- 建议: DROP. No time-series data-selection benchmark exists in this literature. Synthetic KernelSynth series compound the problem because the pool is not even a recognized real dataset.
- 原因: Zero of the 31 papers touch time series, and kernel_synth appears nowhere. It is doubly weak: (1) no modality precedent for data selection, and (2) a synthetic generator rather than an accepted benchmark. Text-perplexity over serialized synthetic numbers measures format memorization, not transferable selection, and is indefensible against this baseline set.

## 最终建议
DEFENSIBLE CORE (keep, 3 modalities): (1) language = FineWeb-Edu, (2) math = FineMath-4plus (OpenWebMath as backup), (3) code = swap codeparrot-clean-valid -> Stack-Edu / StarCoderData Python (or The Stack v2 Python). These three are the only modalities with strong, multi-paper, top-venue selection pools, and crucially they form a coherent family: FineWeb-Edu, FineMath, and Stack-Edu/StarCoderData are all used together by SmolLM2 (and individually by FineWeb NeurIPS 2024, OpenWebMath/DeepSeekMath, The Stack/StarCoder2/phi-1). A reviewer cannot challenge this triplet, and it cleanly supports a held-out-perplexity, modality-agnostic claim across three genuinely distinct text-native modalities.

FIX REQUIRED: code is the one easy win. codeparrot has zero standing in this literature; replacing it with a public Stack-Edu/StarCoderData Python split restores the original, correct 'Python-Edu = code analogue of FineWeb-Edu' design and makes all three core pools come from the same recognized quality-filtered lineage.

DROP (3 modalities): vision-language (COCO-caption text), table (mstz/adult), and timeseries (chronos kernel_synth) should all be removed from the headline 'modality-agnostic data selection' claim.
- vision-language: INDEFENSIBLE as text-only. The entire VL selection literature is image-text PAIRS with CLIP/VQA eval; caption text is just English text wearing a VL label. If multimodality must be claimed, run a real DataComp or LLaVA-665K track under its native protocol as separate work, not as serialized perplexity.
- table and timeseries: there is NO data-selection benchmark for either modality anywhere in the 31-paper corpus, and the specific choices (Adult; synthetic KernelSynth) appear in zero papers. They give reviewers no comparison anchor and read as ad hoc; KernelSynth is additionally synthetic. Keeping them weakens, not strengthens, the paper.

BOTTOM LINE: Reframe the contribution as 'modality-agnostic across text-native modalities' demonstrated on language/math/code (FineWeb-Edu / FineMath-4plus / Stack-Edu), all defensible and lineage-consistent. Treat table, timeseries, and serialized vision-language as out-of-scope for the perplexity pipeline; mention them only as future work or run them under their real, recognized protocols in a clearly separated section."