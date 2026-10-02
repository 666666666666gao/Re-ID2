# DeMo Dual-Axis Re-ID

Research prototype based on the MIT-licensed [official DeMo implementation](https://github.com/924973292/DeMo).

The first comparison uses the same active parameters and 5,632-dimensional descriptors:

- `ordinary`: complete DeMo PIFE/HDM/ATMoE plus a frequency branch. Additional capacity refines frequency evidence only; the outputs are concatenated.
- `dual`: the same seven HDM relation modules are reused on three spatial frequency bands. Modality-preserving frequency evidence and relation evidence exchange conditional messages and use a joint 7-by-3 router.
- `demo`: complete DeMo reference with its original 5,120-dimensional descriptor.

Frequency bands are applied to the 2D patch grid, with fixed radial boundaries of 0.125 and 0.25 cycles per patch. Three band modules are reused across modalities; 21 separate experts are not constructed. The interaction term permits non-factorizable routing. Outer sigmoid gates can activate both experts.

Run `run_experiment.py --help` for commands. Each experiment starts from public CLIP ViT-B/16, trains for 50 epochs, and selects the earliest maximum development mAP checkpoint. The initial seed42 comparison is extended to seeds42/43/44 for all three models on RGBNT201, RGBNT100 and MSVR310:27 training runs. Identity-disjoint development splits are fixed in `splits.json`; training uses the fit subsets, not the entire official training sets. Contributions are not supervised in this first comparison.

`launch_repeats.py` continues each GPU's existing queue. `launch_full_evaluation.py` evaluates the fixed best checkpoints after that GPU's training finishes. `full_evaluation.py` reports development and official-test mAP, mINP, Rank1/5/10/20, CMC1..50 and per-query/group records. Official test never chooses checkpoints. `collect_full_suite.py` audits all50 epochs, paired batch sequences, actual optimizer updates and AMP skips, then publishes every seed's metrics, sample standard deviations and paired query harms/rescues. Current completion status and limitations are recorded in the single handoff below; planned evaluations are not completed results.

One consolidated experiment handoff is maintained at `docs/实验交接.md` and mirrored byte-for-byte to the user's Desktop/document and both active servers. Data, pretrained weights, research checkpoints and private connection details are not committed.
