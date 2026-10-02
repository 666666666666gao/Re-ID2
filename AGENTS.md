# DeMo Dual-Axis experiment

Keep changes minimal. Preserve original DeMo modules except the evidenced hardcoded pretrained path fix. Do not modify the sibling TriFusion project or its archived experiments.

This is the new user-authorized first comparison: complete DeMo plus ordinary frequency versus dual-axis plus joint routing, with the same active parameters and descriptor dimension. One complete DeMo reference per dataset is included. Use existing pretrained CLIP, seed42, B64 and 50 epochs. No contribution loss, tuning sweep or official-test checkpoint selection in this first stage.

Use the existing conda environments on servers2025/2026. Servers2027/2028 currently have occupied GPUs. Avoid their tasks. Follow the existing dataset ground-truth filtering: same-ID/same-camera for RGBNT, same-ID/same-scene for MSVR310.

Keep only one consolidated handoff at docs/实验交接.md. Mirror the same bytes to C:/Users/gb/Desktop/document/DeMo双轴实验交接.md and each active remote project's docs directory. Push authorized code and text results to https://github.com/666666666666gao/Re-ID2. Never publish credentials, private connection configuration, dataset images or weight binaries.

Review correctness with the experiment-bridge skill's fresh Codex reviewer before launching. Poll at expected milestones or 240 seconds. Do not restart a running experiment just because observation times out.
