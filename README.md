# DeMo Dual-Axis Re-ID

Research prototype based on the MIT-licensed [official DeMo implementation](https://github.com/924973292/DeMo).

The first comparison uses the same active parameters and 5,632-dimensional descriptors:

- `ordinary`: complete DeMo PIFE/HDM/ATMoE plus a frequency branch. Additional capacity refines frequency evidence only; the outputs are concatenated.
- `dual`: the same seven HDM relation modules are reused on three spatial frequency bands. Modality-preserving frequency evidence and relation evidence exchange conditional messages and use a joint 7-by-3 router.
- `demo`: complete DeMo reference with its original 5,120-dimensional descriptor.

Frequency bands are applied to the 2D patch grid, with fixed radial boundaries of 0.125 and 0.25 cycles per patch. Three band modules are reused across modalities; 21 separate experts are not constructed. The interaction term permits non-factorizable routing. Outer sigmoid gates can activate both experts.

Run `run_experiment.py --help` for commands. Each experiment starts from public CLIP ViT-B/16, trains for 50 epochs at seed 42, and selects the earliest maximum development mAP checkpoint. Identity-disjoint development splits are fixed in `splits.json`; these are development experiments, not full-training benchmark reproductions. Contributions are not supervised in this first comparison.

One consolidated experiment handoff is maintained at `docs/实验交接.md` and mirrored byte-for-byte to the user's Desktop/document and both active servers. Data, pretrained weights, research checkpoints and private connection details are not committed.
