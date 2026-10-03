已完成审查并写入 [审查 JSON](C:/Users/gb/projects/demo_dual_axis_20261002/results/preflight/axis_collaboration_v4_first5_integrity_audit.json)。总体 **WARN**；A/B/C/D/F PASS，E WARN；无证据完整性阻塞项。

标准库实际复算 1,249 项全部通过：250 轮、12,723 次更新、0 AMP 跳过；14 张表/14,145 query/595 分组，六指标最大误差 1.42e-14，CMC 与逐 query INP 误差0。

三项 V4 双轴的开发 mAP/Rank-1 同时+2均未达成。仅支持单seed开发与冻结干预结论；未复验GPU、远端原始数组/权重，也未重算旧V1的351条件。机器 trace 和输入SHA已保存，未修改源码或唯一交接文档。
