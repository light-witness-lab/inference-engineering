# 单机 P/D 分离实验评审 · Single-Node Prefill/Decode Disaggregation

> 状态：方案评审（2026-11-01），未开始实测。
> 实验机：Mac Studio (M1 Max, 64 GB 统一内存, macOS 15.7.3)。
> 上游可行性调研（DeepInfer/SGLang/Mooncake/llama.cpp issue 取证，2026-10-27~30）在私有学习仓
> `bestksl/Learn001` 的 `docs/mac-pd-options.md`；本文只保留实验需要的结论，证据在彼处可溯。
> Evidence levels: `verified`（本机实测后使用）· `loaded-only`（仅读到官方资料/他人实测）·
> `not-verified`（假说，待本实验证伪）。

## 1. 问题：P:D 比例在单机上调节，有没有收益？

P/D 分离把 prefill（compute-bound）与 decode（memory-bandwidth-bound）拆到独立实例池，
再按负载调节两池比例（如 1P:2D）。集群里比例调节成立，靠三个前提：

1. **独立物理资源池**：D 池多一个副本就多一张卡/一台机；
2. **独立扩缩**：长输出（agent/对话）负载向 D 倾斜时只加 D，不为用不到的 prefill 算力付钱；
3. **干扰消除**：大批 prefill 不再打断在跑的 decode step（TBT SLO）。
   参照系：DeepSeek DeepInfer 官方 QA 给出其线上配比——prefill EP2–8、decode EP64–128，
   GPU 数约 1:1（随工作负载而变，不是通用最优比）。

**单颗 M1 Max 上，前提 1 不成立**：开 1 个 P 进程 + 2 个 D 进程不等于三块资源，只是三个进程
时间片复用同一颗 GPU。由此有三个硬性推论（`loaded-only`，待实测转 `verified`）：

- GPU 总 FLOPs 不变，prefill 总量也没有减少；多进程只增加调度与上下文切换开销。
- **模型权重按进程复制**（llama.cpp issue #21266，ggerganov 2026-04-02 确认不可避免）：
  14B Q4 约 9 GB，1P:2D 就是约 27 GB 权重，挤占 KV cache 余量。
- 统一引擎的 **chunked prefill** 已经把「prefill 打断 decode」这个最大痛点缓解掉大半，
  静态切分反而损失统计复用（P 空闲时它的 GPU 份额不能让给 D）。

但有三条**仍可能产生真实收益的假说**，值得实测，不做无端否定：

| 假说 | 机制 | 预判 |
| --- | --- | --- |
| H1 异质器件分工 | P 走 CPU（M1 Max 的 AMX 矩阵单元对 prefill 不弱），D 独占 GPU；两类算力物理分开，decode 永不被抢 | **最可能成立**，但 CPU prefill 可能比 GPU 慢到抵消收益 |
| H2 干扰隔离 | 即使同在 GPU，进程边界可能压住并发场景的 TBT 尾延迟 | 存疑：Metal 不提供这类抢占 QoS，`not-verified` |
| H3 比例适配 | 1:2 的 D 池在多用户长输出下改善人均吞吐 | 大概率被统计复用反杀；测了才知道 |

实验的产出目标是**用数据证实或证伪 H1–H3**，而不是证明「单机 PD 有收益」。
全部被证伪同样是有价值的工程结论（与 [on-device CPU inference playbook](../on-device-cpu-inference-playbook.md)
第 0 节同构：先重塑问题，再调 kernel）。

## 2. 方案评审

| # | 方案 | 形态（本机） | 现状 | 真 KV 迁移 | 工作量 | 检验对象 | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 单体 llama-server | 一个 Metal 进程，chunked prefill | 正式版 | — | 极低 | 基线 | **必做基线** |
| A1 | llama.cpp slot 文件交接 | 双 Metal 进程：P 端 `/slots save` → 共享目录 → D 端 `/slots restore` | 正式版（端点已发布，loaded-only→待验） | ✅ 顺序文件 | 小（≤200 行 router） | 协议开销、H2/H3 | **首选** |
| A2 | llama.cpp 异质分工 | P 端 `-ngl 0` 纯 CPU / D 端全 Metal，交接同 A1 | 正式版能力组合 | ✅（跨后端恢复 **not-verified**，首要排雷点） | 中 | **H1** | **重点** |
| A3 | llama.cpp RPC 流式 PD | 编译 draft PR #24901 分支，KV 边算边传 | 未合并的 draft，loopback 已验证 bitwise 一致（loaded-only） | ✅ 流式 | 中高（自编译分支） | 生产形态预演 | 尝鲜，靠后 |
| A4 | mlx-lm prompt cache | 双 Python 进程，`save/load_prompt_cache`（safetensors） | 正式版 API | ✅ 顺序文件 | 中（全自己写） | 结论可复现性/Python 栈 | 可选平行验证 |
| B1 | SGLang-MLX 双实例 | 两个 `SGLANG_USE_MLX=1` server | 可用但 experimental | ❌ 无 connector（Mooncake/NIXL 均 Linux-only） | 小 | 「假 PD」对照 | 只做对照 |
| B2 | Ollama | llama.cpp 的封装 | — | 不暴露 slot 状态 API | — | — | 否决 |
| B3 | vLLM CPU on macOS | — | Mac 无 MPS 后端（#22629 not_planned），CPU 路径非主力 | — | — | 否决 |
| B4 | LMDeploy / TensorRT-LLM | — | CUDA-only | — | — | 否决 |
| B5 | 纯网关分流（LiteLLM 类） | 请求按长度路由到两个普通后端 | 现成 | ❌ D 端拿不到 KV，必须重算 | 小 | 量化「没有 KV 迁移的分流浪费多少」 | 对照基线 |
| C1 | 第二台机器（Mac/Mac mini + 10GbE） | A1 交接走网络，比例调节首次获得真实意义 | 依赖硬件 | ✅ | 中 | 前提 1 成立后的比例实验 | 下一阶段 |
| C2 | ANE/CoreML 充当 P | — | llama.cpp CoreML 覆盖有限，`not-verified` | 未知 | 高 | 后排 |

评审结论：**A0 → A1 → A2 为主线**（同一份 llama.cpp 构建即可完成），B1/B5 做对照，
A3/A4 选做，C1 等第二台设备。SGLang/vLLM/Mooncake 这一族在 macOS 没有真 P/D，已在调研中排除。

## 3. 实验序列与测量设计

公共设置：同一模型 `DeepSeek-R1-Distill-Qwen-14B` Q4（GGUF，与 Learn001 M2a 共用），
统一 context 参数；温度 temp=0 做正确性 sanity；工作目录用共享文件夹（同机免网络变量）。

- **E0 基线（A0）**：单 Metal server。prompt 长度 512 / 2k / 8k / 32k，并发 1/4/8 扫点。
  记录 TTFT、TBT p50/p99、decode tok/s、峰值内存、权重占用。
- **E1 协议跑通（A1）**：双 Metal server + router。先单用户 8k prompt 验证 D 端确实不重算
  （TTFT 对比、生成文本与 A0 一致性）；再量 KV blob 字节数/token、`save_ms`、`restore_ms`、
  序列化在总时延占比；最后跑 1:1 与 **1:2** 两种进程配比的并发扫点，检验 H2/H3。
- **E2 异质分工（A2）**：P=`-ngl 0` CPU、D=Metal。**第一步先排雷**：CPU 后端 save 的状态在
  Metal 后端 restore 后 temp=0 输出是否一致（跨后端布局兼容性，`not-verified`）。
  通过后重复 E1 扫点，与 A0/A1 对比——H1 的成立与否在此见分晓。
- **E3 假 P/D 对照（B1 或 B5）**：同样路由、但不做 restore，D 端重算全 prompt。
  与 E1 的 TTFT 差值即「KV 迁移省下来的 prefill」，与 save/restore 开销对照，得出单机净收益平衡点
  （预期：短 prompt 净亏、长 prompt 才可能净赚，求拐点长度）。
- **E4（可选）**：A3 draft 分支观察流式交接对拐点的影响；A4 用 mlx-lm 复现 E1 关键数字。

每个实验记录：llama.cpp 版本 commit、模型文件与 SHA-256、启动参数全量、原始 JSON 计时
（slot 端点返回 `save_ms/restore_ms/n_written`），表格入本目录后续 `results-*.md`。

## 4. 边界与非目标

- 单机实验回答的是「云时代 P/D 技术在单 SoC/单主机的成立条件」，**数字不外推到手机 SoC**
  （street-snap 目标机无 AMX、无统一内存 64 GB），与仓库 README 既有纪律一致。
- 不追求「单机性能提升」这一预设结论；H1–H3 被证伪即收尾写负结果。
- 不引入 Mooncake/NIXL/RDMA（Linux 生态），不碰 V4.1-Flash（见 Learn001 兼容性判定）。
- 同机实验的天然局限：共享一颗 GPU，配比实验的价值主要是**确认静态切分的代价**；
- 真要研究 1:2 这类比例，正路是 C1（第二台设备）或云端双卡（Learn001 M4）。

## 5. 与仓库其他工作的关系

- 方法论沿用 [on-device-cpu-inference-playbook](../on-device-cpu-inference-playbook.md)：
  先重塑问题、硬截止、测量优先；本实验是「云技术设备侧成立条件」在生成式 LLM 上的延伸。
- 若 E2（CPU 当 P）成立，对 street-snap 类设备的启发是「廉价常在线支路 vs 稀疏重支路」之外的
  另一种分工范式，但只作参考，不直接进入 App 方案。
