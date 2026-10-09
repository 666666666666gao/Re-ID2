# DeMo Dual-Axis experiment

## Latest missing-evaluation override — 2026-10-09

For new missing-modality evaluations, use only the six DeMo paper settings: missing R, N, T, RN, RT or NT, applied equally to query and gallery. Use every official query/gallery record and the fixed normal-input mAP-best checkpoint. Do not launch new 49-pair evaluations; keep the completed historical 49-pair evidence. Report all six metrics for each condition and the equal-condition mean, with negative cases retained. DeMo reports these missing settings for RGBNT201/RGBNT100; applying them to MSVR310 is our explicit extension.

Keep the best K RGBNT201 checkpoint and its published evidence. M normal training, the paper-six queue and selected training-gradient diagnostics are closed; never restart their completed owners. Finish normal-input experiments before the six missing settings. New neural work requires fresh source review and real preflight. All neural work remains on 2026 physical GPU2/3, serialized per card, at most two models.

## Latest user override — 2026-10-04

For all new experiments use the entire official training split and every official query/gallery record. Do not create or use an artificial fit/dev identity holdout. Historical subset experiments remain archived evidence. The full-data baseline stage follows the original highest benchmark mAP checkpoint principle, with earliest ties shared across models, and discloses benchmark-based selection. Missing evaluations use that fixed checkpoint, the six symmetric DeMo paper settings specified above, and original GT camera/scene exclusion. All neural jobs run only on 2026 physical GPU2/3, serialized per card, at most two concurrent. No temperature or power conditions. Rebuild complete DeMo and same-augmentation availability-correct DeMo references before advancing the pending M3b control-head experiment. Fresh source review and real preflight remain required.

The older first-stage instructions below are historical where superseded by this user override.


Keep changes minimal. Preserve original DeMo modules except the evidenced hardcoded pretrained path fix. Do not modify the sibling TriFusion project or its archived experiments.

This is the new user-authorized first comparison: complete DeMo plus ordinary frequency versus dual-axis plus joint routing, with the same active parameters and descriptor dimension. One complete DeMo reference per dataset is included. Use existing pretrained CLIP, seed42, B64 and 50 epochs. No contribution loss, tuning sweep or official-test checkpoint selection in this first stage.

Use the existing conda environments on servers2025/2026. Servers2027/2028 currently have occupied GPUs. Avoid their tasks. Follow the existing dataset ground-truth filtering: same-ID/same-camera for RGBNT, same-ID/same-scene for MSVR310.

Keep only one consolidated handoff at docs/实验交接.md. Mirror the same bytes to C:/Users/gb/Desktop/document/DeMo双轴实验交接.md and each active remote project's docs directory. Push authorized code and text results to https://github.com/666666666666gao/Re-ID2. Never publish credentials, private connection configuration, dataset images or weight binaries.

Review correctness with the experiment-bridge skill's fresh Codex reviewer before launching. Poll at expected milestones or 240 seconds. Do not restart a running experiment just because observation times out.
