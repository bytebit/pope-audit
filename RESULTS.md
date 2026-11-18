# POPE 基线结果（修正版）

**模型**：LLaVA-1.5-7B（llava-hf/llava-1.5-7b-hf），4bit NF4 量化（bitsandbytes）
**硬件**：RTX 2060 12G 单卡，峰值显存约 3.8 GiB
**评测**：POPE 三子集 × 3000 条，greedy 解码，max_new_tokens=8，批量 size=8（按问题长度分组、零填充）
**运行时间**：每子集约 26 分钟，总约 1.3 小时

## 最终指标

| 子集 | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| adversarial | 0.7697 | 0.7146 | 0.8980 | 0.7959 |
| popular | 0.8383 | 0.8015 | 0.8993 | 0.8476 |
| random | 0.8847 | 0.8737 | 0.8993 | 0.8863 |
| **平均** | **0.8309** | 0.7966 | 0.8989 | **0.8433** |

## 与论文基线对照

| 配置 | Acc | F1 |
| --- | --- | --- |
| 本文复现（4bit，12G） | 0.8309 | 0.8433 |
| 论文 Regular（TopicA 精读引述，fp16） | 0.7983 | 0.7929 |

差异来源（预期内）：① 评测协议不同（短生成+首词匹配 vs 原版长生成协议）；② HF 检查点与原版权重存在微小差异；③ 4bit 量化。趋势一致（adversarial 最差、random 最好），可作为本硬件环境下的可复现基线。

## 已知问题与修复记录

**问题**：transformers 5.9 + 4bit 下，带左填充（padding）的批量推理在部分行产生 NaN logits，导致约 2.9% 样本回答为空（生成 token 全为 0）。
**定位**：NaN 出现在 LLM 深层（第 16-31 层），与左填充量相关；同一批中零填充的行全部正常；逐条推理全部正常。
**修复**：`--group-by-length` 按问题 token 长度分组批量，保证批内零填充。验证：此前出问题的 78 条全部恢复，其余样本与旧结果逐字一致。
**产出**：修正版结果 `outputs/pope_baseline_20260828_210602.json`（含逐条预测）；旧版（含 NaN 空回答）文件保留作诊断。

## 复现命令

```powershell
$env:HF_HOME="E:\dev_projects\PAPERS\paper3\exp_topicA\hf_cache"
python scripts/eval_pope.py --subsets adversarial popular random --batch-size 8 --group-by-length
```

---

# ResDec 缓解复现（2026-08-29）

**协议对齐结论（重要）**：论文描述 POPE 时明确说 prompt 附加 "Please answer yes or no."（附录 A.1），且附录 C.1 给出默认历史窗口 **W=8**（不是 15）。只有同时满足这两个条件，才能复现出论文的 +7 增益；plain 模板下 ResDec 只涨 +1~2 点。

## 复现配置

- prompt：`"{question} Please answer yes or no."`（ins-yesno 模板）
- ResDec：α=0.5，β=0.1，K=256，**W=8**，valley=argmin，greedy
- 模型/硬件同基线：LLaVA-1.5-7B 4bit NF4，RTX 2060 12G，批量 8（按长度分组、零填充）
- 指标口径：首词解析（`startswith("yes")`），与 Regular/ResDec 一致
- 结果文件：`outputs/pope_baseline_20260829_001941.json`、`outputs/pope_resdec_20260829_015107.json`

## 结果对照（Acc/F1）

| 子集 | Regular 复现 | Regular 论文 | ResDec 复现 | ResDec 论文 | 增益(复现) | 增益(论文) |
| --- | --- | --- | --- | --- | --- | --- |
| adversarial | 76.13 / 72.27 | 76.03 / 76.26 | 82.67 / 82.35 | 84.50 / 83.69 | +6.53 | +8.47 |
| popular | 78.23 / 73.85 | 79.98 / 79.34 | 85.80 / 85.07 | 86.97 / 87.22 | +7.57 | +6.99 |
| random | 80.27 / 76.01 | 83.49 / 82.28 | 88.57 / 87.62 | 90.21 / 89.87 | +8.30 | +6.72 |
| **平均** | **78.21 / 74.04** | 79.83 / 79.29 | **85.68 / 85.01** | 87.23 / 86.93 | **+7.47** | +7.40 |

**结论**：增益幅度精确复现（+7.47 vs +7.40 acc）。绝对分数低约 1.5~2 点，归因于 4bit 量化 + HF 转换检查点 + 评测口径差异，趋势（adversarial 最差、random 最好）与论文一致。

## 重要发现：增益主要来自"模板回显伪影"修复，而非真实幻觉修正

在 ins-yesno 模板下，Regular 基线约 **23%~29%** 的回答出现 `ASSISTANT:` 回显（如 `. assistant: yes`）：

| 子集 | 回显总数 | 因回显答错 | 真实答错 | 回显占错误比例 |
| --- | --- | --- | --- | --- |
| adversarial | 744/3000 | 334 | 382 | 47% |
| popular | 864/3000 | 339 | 314 | 52% |
| random | 693/3000 | 329 | 263 | 56% |

ResDec 的修复分解（相对 Regular 的翻转）：

| 子集 | 修回显 | 修真实 | 破坏回显 | 破坏真实 | 净翻转 |
| --- | --- | --- | --- | --- | --- |
| adversarial | 257 | 27 | 41 | 47 | +196 |
| popular | 260 | 34 | 34 | 33 | +227 |
| random | 250 | 28 | 9 | 20 | +249 |

**解读**：
1. ResDec 的 +7.5 增益约九成来自把 `. assistant: yes` 这类回显拉回 `yes`；真正把"该说 no 却说了 yes"的对象幻觉修正掉的部分只有 ~30 条/子集，且被 20~47 条真实破坏抵消（adversarial 上真实净效应为负）。
2. plain 模板（无回显）下 ResDec 只有 +1~2 点，且那些翻转才是真实幻觉修正（300 条样本上 fix 7 / break 2）。
3. 结论：论文在 POPE 上宣称的 +7 缓解，机制上主要是"修复指令性 prompt 引发的模板回显"，不是纯视觉幻觉抑制。这既是复现要点，也是选题审计（题目 2）的核心素材。

## 评测口径敏感性（短生成 + 解析规则）

| 子集 | Reg/首词 | Reg/官方解析 | Res/首词 | Res/官方解析 |
| --- | --- | --- | --- | --- |
| adversarial | 76.13/72.27 | 73.60/76.19 | 82.67/82.35 | 49.97/66.62 |
| popular | 78.23/73.85 | 72.03/75.04 | 85.80/85.07 | 49.93/66.61 |
| random | 80.27/76.01 | 79.37/80.37 | 88.57/87.62 | 49.97/66.62 |

注：官方解析（LLaVA eval_pope.py：取首句、去逗号、按词找 no/not）对 ResDec 的输出 `no\nis there...` 会把 "no" 与下一词粘连而误判为 yes，导致指标崩塌。已用 max_new_tokens=16 验证这不是短生成伪影：ResDec + ins-yesno 模板的输出稳定为 `yes\n...` / `no\n...`（首词 + 换行 + 追问式续写），"no" 永远以 "no\n" 形式粘连出现。因此官方解析与该模板下的 ResDec 不兼容，评测必须统一首词解析；这正是"协议敏感性"的又一例证。

## 复现命令

```powershell
# Regular（ins-yesno）
python scripts/eval_pope.py --subsets adversarial popular random --batch-size 8 --group-by-length --prompt-template ins-yesno

# ResDec（ins-yesno, W=8）
python scripts/eval_pope_resdec.py --subsets adversarial popular random --batch-size 8 --window 8 --prompt-template ins-yesno

# 结果对照/分解分析
python scripts/analyze_resdec_full.py
```

---

# PAS 检测基线（2026-08-29）

**数据**：POPE 的 500 张唯一 COCO val2014 图（9000 条 POPE 问题 → 500 个 COCO id，映射经官方 COCO 原图 md5 校验 8/8 通过）。对象真值来自 `instances_val2014.json`（80 类），平均 3.88 对象/图。

**协议**：prompt "Please help me describe the image in detail."，greedy，layer-0 多头注意力捕获；CHAIR 式最长匹配找对象 token（已修复 "teddy bear" 子串误匹配 "bear" 的 bug）；检测器 PAS / SVAR / NLL / Entropy；标签 = 对象是否在图内 GT。

## 检测结果（AUROC）

| 检测器 | 64 token (n=1793, neg 12%) | 128 token (n=2985, neg 21%) | +10% 标签噪声 | +30% 标签噪声 |
| --- | --- | --- | --- | --- |
| PAS（prelim 注意力） | 0.480 | 0.412 | 0.444 | 0.471 |
| SVAR（图像注意力） | 0.520 | 0.588 | 0.556 | 0.529 |
| NLL | 0.496 | 0.518 | 0.507 | 0.496 |
| Entropy | 0.492 | 0.515 | 0.505 | 0.495 |

**解读**：
1. 生成长度决定检测器是否有效：64 token 全部≈随机；128 token 时 SVAR 才有真实信号（0.588）。
2. **PAS 方向与论文相反**（0.412 < 0.5）：在 4bit 复现下"幻觉 token 更依赖 prelim"不成立，甚至反向。量化/规模/prompt 可能翻转机制。
3. 标签噪声把全部检测器压回≈0.5，且检测器排序随噪声水平变化——"哪个检测器最好"不是稳定结论。
4. 结论：检测协议和缓解协议一样脆弱；论文必须报告完整协议（生成长度、GT 来源、量化、匹配规则），否则检测结论不可比。

## 原版协议验证（5000 图 × 512 token cap，2026-08-29）

**数据**：COCO val2014 随机采样 5000 图（seed 0），本地 500 POPE 图复用 + 4948 张下载（md5 校验 8/8），对象真值来自 instances_val2014.json。生成长度实测：中位 134、p90 157、最大 198——都在 EOS 处停止，未触及 512 cap。

**结果**（26,761 个对象 token，正 21,459 / 负 5,302，负占比 19.8%）：

| 检测器 | 本文 @5000×512 | 论文（LLaVA-1.5 fp16, MSCOCO, greedy） | Δ |
| --- | --- | --- | --- |
| PAS | **0.387** | 0.842 | **-0.455（方向相反）** |
| SVAR | 0.613 | 0.815 | -0.202 |
| NLL | 0.489 | 0.565 | -0.076 |
| Entropy | 0.487 | 0.716 | -0.229 |

标签噪声（5 seed 均值）：SVAR 0.613→0.575(10%)→0.532(30%)；PAS 0.387→0.425→0.468（从下方趋近随机）；NLL/Entropy 全程 ≈0.49-0.50。

**结论**：即使按原版协议规模跑，PAS 论文的检测结论在我们的 4bit 环境仍无法复现，PAS 方向稳定为负（幻觉 token 反而更少注意 prelim），SVAR（图像注意力）是唯一可用信号。剩余差异来自 4bit 量化 + HF 检查点（12G 显存无法跑 fp16）。这本身就是审计结论：**检测信号在 4bit 部署下可反转，而 4bit 正是消费级 GPU 的主流用法**。

结果文件：`outputs/pas_detection_20260829_172214.json`、`outputs/pas_5000_summary.json`；分析：`scripts/analyze_pas_5000.py`。

## 量化对照：8bit（2026-09-15）

| 检测器 | 4bit（500 图/128 token） | 8bit（同协议） | 论文（fp16） |
| --- | --- | --- | --- |
| PAS | 0.412 | **0.413** | 0.842 |
| SVAR | 0.588 | **0.587** | 0.815 |
| NLL | 0.518 | 0.521 | 0.565 |
| Entropy | 0.515 | 0.518 | 0.716 |

**结论**：8bit 与 4bit 几乎完全一致（差异 <0.005）→ **PAS 反转不是量化伪影**。差异只能来自 HF 检查点/实现细节或评测协议本身。结果文件：`outputs/pas_detection_20260915_195610.json`（8bit）。

## 注意力分解（500 图 × 128 token，6 层，2026-09-15）

把 PAS 的注意力行按"看到的是谁"拆开（对象 token 对各类历史位置的注意力，头平均）：

| 层 | PAS_all（全部文本） | PAS_gen（仅生成 token） | PAS_inst（仅指令） | SVAR（图像） |
| --- | --- | --- | --- | --- |
| 0 | 0.4116 | 0.3944 | **0.6745** | 0.5884 |
| 4 | 0.5321 | 0.5340 | 0.3502 | 0.4680 |
| 8 | 0.3850 | 0.3776 | 0.5488 | 0.6150 |
| 16 | 0.3627 | 0.3595 | 0.4610 | **0.6374** |
| 24 | 0.3989 | 0.3995 | 0.4582 | 0.6011 |
| 31 | 0.5206 | 0.5144 | 0.5932 | 0.4794 |

**结论**：
1. 论文定义的 PAS 量（对 prelim/生成 token 的注意力）在各层都 <0.5——与论文主张方向相反且稳健。
2. 真正有判别力的信号在别处：**层 0 的"指令注意力"0.6745**、**层 16 的"图像注意力"0.6374**。
3. 机制解读：在我们的设置里，区分幻觉 token 的"语言先验"信号挂在 **prompt 指令 token** 上（问题的语言模式），而不是此前生成的输出 token 上。这解释了为什么 PAS 的具体形式失效、而相邻的注意力统计量有效。

结果文件：`outputs/pas_variants_20260915_203552.json`（500 图）；`outputs/pas_variants_20260915_200531.json`（100 图，同结论）。

## 官方实现核对（2026-09-21）

拉取 PAS 官方代码（`github.com/lanl/pas`，文件 `scripts/compute_scores.py`、`src/pas/utils/internal.py`）逐行核对后发现两处约定差异：

1. **注意力行取"对象 token 之前那个位置"**（代码注释：*all token positions refer to the position BEFORE the object*），我们原先取的是对象 token 自身的行；
2. prelim 的键范围是 `prompt_end_idx:`（生成段），我们原先也涵盖了指令段（另有 pas_gen 变体单独测过）。

按官方约定在 500 图上重跑：

| 层 | PAS_official | PAS_official（全部文本键） | 我们原实现（对象行） |
| --- | --- | --- | --- |
| 0 | 0.3466 | 0.3730 | 0.3944 |
| 8 | 0.3770 | 0.3938 | 0.3776 |
| 16 | 0.3312 | 0.3329 | 0.3595 |
| 31 | 0.5134 | 0.5260 | 0.5144 |

**结论**：官方约定下反转**更强**（层 0 为 0.347，论文报 0.842）——所以反转不是我们的实现偏差，而是该检测信号在公开 LLaVA-1.5-7B 管线上无法复现。官方代码同样计算 instruction / image attention（方向与我们发现的有效信号一致），但论文头条结论用的是 prelim 量。

结果文件：`outputs/pas_variants_20260921_192719.json`。

---

# 第二模型：Qwen2.5-VL-7B 回显验证（2026-09-15）

**动机**：检验 `ASSISTANT:` 回显伪影是"VLM + yes/no 指令"的普遍现象，还是 LLaVA 特有。

## 结果（4bit，500 条/子集，batch 1）

| 子集 | prompt | Regular Acc/F1 | VCD Acc/F1 | VCD 增益 [95% CI] | 回显率 | 翻转分解（修回显/真修正/真破坏） |
| --- | --- | --- | --- | --- | --- | --- |
| random | plain | 87.80/86.29 | 88.60/87.19 | +0.80 [−0.20,+2.00] | **0/500** | 0 / 6 / 2（净 +4） |
| random | ins-yesno | 89.80/88.79 | 89.80/88.84 | 0.00 [−1.00,+1.00] | **0/500** | 0 / 3 / 3（净 0） |
| adversarial | plain | 87.20/85.71 | 87.00/85.46 | −0.20 [−1.60,+1.20] | **0/500** | 0 / 5 / 6（净 −1） |
| adversarial | ins-yesno | 88.60/87.64 | 87.40/86.39 | −1.20 [−2.80,+0.20] | **0/500** | 0 / 4 / 10（净 −6） |

**结论**：
1. **Qwen2.5-VL 完全没有回显**（4 组 × 500 条全为 0），输出形态干净（`yes.` / `no` / `yes, there is a ...`），ins-yesno 指令不压低基线（random 上反而 +2.0）。
2. **在没有伪影的模型上，VCD 的真实效应为零**：增益 −1.2 ~ +0.8，置信区间全部包含 0，净效应 −6 ~ +4 条/500。
3. 回显伪影是 **LLaVA-1.5 特有的 model×template 交互**，不是基准或模板本身的普遍性质。因此 POPE 上的"缓解增益"不能跨模型比较。
4. Qwen 本身在 POPE 上更强（87–90% vs LLaVA 76–88%），进一步说明跨模型的"增益"比较没有意义。

结果文件：`outputs/qwen_pope_{random,adversarial}_20260915_205435.json`；分析：`scripts/analyze_qwen_results.py`。

## 缓解侧的精度对照：LLaVA 8bit 回显率（2026-09-15）

| LLaVA-1.5 random 前 500 条，ins-yesno | Acc | F1 | 回显率 |
| --- | --- | --- | --- |
| 4bit NF4 | 79.80 | 75.43 | 120/500（24.0%） |
| 8bit | 72.40 | 63.30 | **212/500（42.4%）** |

**结论**：8bit 下回显更严重（42.4% vs 24.0%），准确率更低。回显是模型指令跟随行为（精度越高越忠实复现模板表面形式），不是量化噪声。这也意味着缓解方法的"增益"随伪影率而非幻觉水平变化。

---

# 配对显著性检验与 CHAIR 自由生成对照（2026-09-21）

## McNemar 配对检验（总体增益 vs 真实修正）

| 方法/子集 | prompt | b/c（不一致对） | p（总体） | p（仅真实） |
| --- | --- | --- | --- | --- |
| ResDec / adversarial | ins-yesno | 284/88 | 3.5e-25 | 0.027（方向为负） |
| ResDec / popular | ins-yesno | 294/67 | 4.9e-35 | 1.00 |
| ResDec / random | ins-yesno | 278/29 | 3.3e-52 | 0.31 |
| VCD / adversarial | ins-yesno | 269/75 | 8.8e-27 | 0.136 |
| VCD / popular | ins-yesno | 285/75 | 6.1e-30 | 0.41 |
| VCD / random | ins-yesno | 260/37 | 2.0e-42 | 0.56 |
| VCD / 三子集 | plain | 29–42/38–43 | 0.15–0.74 | 同左（无回显） |
| Qwen2.5-VL（两子集） | 两模板 | 3–6/2–10 | ≥0.18 | ≥0.18 |

**结论**：总体增益统计上压倒性显著，去掉伪影后统计上不显著——这是论文主张的最强形式。脚本：`scripts/mcnemar_tests.py`。

## CHAIR 自由生成对照（300 图，128 token，标准 caption prompt）

| 方法 | CHAIR_S ↓ | CHAIR_I ↓ | 提及数/图 | CHAIR_S 变化 | CHAIR_I 增量 [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Regular | 0.6000 | 0.2153 | 5.98 | — | — |
| ResDec | 0.7600 | 0.2858 | 7.16 | +69/−21 图，p=3.9e-7 | **+0.0746** [+0.0509,+0.0992] |
| VCD | 0.7267 | 0.2426 | 7.01 | +56/−18 图，p=1.1e-5 | **+0.0374** [+0.0164,+0.0575] |

**结论**：在自由生成基准上，两个方法都**显著加重**对象幻觉（含幻觉对象的 caption 比例 60.0%→76.0%/72.7%）。绝对 CHAIR 值因 COCO 实例标注不完整而偏高，但三方法同口径可比。这直接回答了审稿人最可能提的问题："那它们在 CHAIR 上到底有没有真效果？"——答案是没有，反而更差。

结果文件：`outputs/chair_20260921_185151.json`；分析：`scripts/analyze_chair.py`。

---

# 模型覆盖扩展：回显伪影的检查点特异性（2026-09-22）

**动机**：把"回显伪影"的模型覆盖从 2 个扩到 4 个，并提供受控对照——**同 chat template、同家族、不同检查点**。

## 关键控制：模板相同

`llava-1.5-7b-hf` 与 `llava-v1.6-vicuna-7b-hf` 的 `chat_template.json` **完全一致**（都以 `ASSISTANT:` 结尾），主干同为 Vicuna-7B。因此两者的差异只能归因于**检查点本身的指令微调**。

## 四模型回显矩阵（POPE，4bit，batch=1）

| 模型（LLM 主干） | 子集 | plain 回显 | ins-yesno 回显 | ins-yesno Acc/F1 |
| --- | --- | --- | --- | --- |
| LLaVA-1.5-7B（Vicuna） | adv/pop/ran | **0.0%** | **24.8 / 28.8 / 23.1%** | 76.13 / 78.23 / 80.27 |
| LLaVA-1.5-7B（Vicuna）8bit | random | – | **42.4%** | 72.40 |
| LLaVA-NeXT-7B（Vicuna） | random | 0.0% | **0.0%** | 88.80 / 87.61 |
| LLaVA-NeXT-7B（Vicuna） | adversarial | 0.0% | **0.0%** | 86.40 / 85.34 |
| LLaVA-OneVision-7B（Qwen2） | random | 0.0% | **0.0%** | 92.80 / 92.31 |
| Qwen2.5-VL-7B（Qwen2.5） | random / adv | 0.0% | **0.0%** | 89.80 / 88.60 |

**结论**：
1. 回显只在 **LLaVA-1.5 这一个检查点**上出现（4bit 23–29%、8bit 42%）；同为 Vicuna 主干、**模板完全相同**的 LLaVA-NeXT 为 0%。
2. 因此伪影不是"VLM + yes/no 指令"的普遍现象，也不是主干决定的，而是**该检查点指令微调的个体特性**。
3. 在 ins-yesno 下基线被压低的也只有 LLaVA-1.5（76–80%）；其余三个模型 86–93%，指令甚至略有帮助（NeXT adversarial +3.2，OneVision random +0.8，Qwen random +2.0）。

**复现坑（新增）**：LLaVA-OneVision 在 `attn_implementation="eager"` + 4bit 下生成全 NaN（输出 token id 全 0，解码成 `!!!!`），换成 **`sdpa`** 后正常。LLaVA-1.5 与 Qwen2.5-VL 在 eager 下无此问题。

结果文件：`outputs/model_llava-next-vicuna-7b_*.json`、`outputs/model_llava-onevision-qwen2-7b_random_*.json`；脚本：`scripts/eval_model_pope.py`、`scripts/show_model_results.py`、`scripts/debug_onevision.py`。

---

# VCD 对照实验（2026-08-30 全量，验证"增益=修回显"的普适性）

**方法**：VCD（Leng et al., CVPR 2024）——原图 logits 与高斯噪声图 logits 对比解码（α=1, β=0.1, σ=0.1，噪声直接加在 CLIP 归一化像素上、每批随机、无 clamp，与论文实现一致），greedy。实现：`scripts/vcd.py`。

## 结果（全量 3000×3，双模板；Regular 复用已有基线）

| 子集/prompt | Regular | VCD | 增益 [95% CI] | fix_echo / fix_genuine / break_genuine |
| --- | --- | --- | --- | --- |
| adversarial / ins-yesno | 76.13/72.27 | 82.60/81.85 | +6.47 [+5.3,+7.6] | 243 / 26 / 39 |
| adversarial / plain | 76.97/79.59 | 77.10/79.64 | +0.13 [−0.5,+0.7] | 0 / 42 / 38 |
| popular / ins-yesno | 78.23/73.85 | 85.23/84.17 | +7.00 [+5.7,+8.2] | 253 / 32 / 40 |
| popular / plain | 83.83/84.76 | 83.50/84.48 | −0.33 [−0.9,+0.2] | 0 / 33 / 43 |
| random / ins-yesno | 80.27/76.01 | 87.70/86.52 | +7.43 [+6.4,+8.5] | 239 / 21 / 26 |
| random / plain | 88.47/88.63 | 88.03/88.23 | −0.43 [−1.0,+0.1] | 0 / 29 / 42 |

**结论**：与 ResDec 完全同构且在全量规模下确认——ins-yesno 下 VCD 的 +6.5~+7.4 增益里 239~253/3000 是修回显，真实修正仅 21~32 条；plain（无回显）下增益全部 ≈0 或为负（真实修正 29~42 被真实破坏 38~43 抵消）。两种机制完全不同的缓解方法（历史残差 vs 对比解码）在 POPE 上的"增益"都主要来自同一个模板伪影。

注：500 条 pilot 曾显示 random ins-yesno 增益 +13.6，全量后为 +7.43——pilot 的 Regular 基线（72.8）恰好偏低，高估了增益。全量数据为准。

结果文件：`outputs/vcd_full_{adversarial,popular,random}_20260830_113621.json`；分析：`scripts/analyze_vcd_full.py`。

## K 敏感性（300 条 ins-yesno，W=8）

K=64: 87.00 | K=128: 87.33 | K=256: 87.33 | K=512: 87.33 → K≥128 完全稳定，论文声称的 64-512 最优区间成立。

## Bootstrap 置信区间（2000 重采样）

- 回显率（Regular/ins-yesno）：adv 24.8% [23.3,26.4]、pop 28.8% [27.2,30.4]、ran 23.1% [21.6,24.5]
- ResDec 真实修正率：0.9% [0.6,1.2]~1.1% [0.8,1.6]，与回显修复率 8.3%~8.7% 区间完全分离
- 脚本：`scripts/bootstrap_ci.py`

## 复现命令

```powershell
python scripts/build_detection_gt.py          # 本地图 → COCO id 映射
python scripts/build_coco_gt.py               # COCO 实例标注 → 每图对象真值
python scripts/verify_image_mapping.py 8      # md5 校验映射（8/8 通过）
python scripts/eval_pas_detection.py --max-new-tokens 128 --batch-size 4
python scripts/analyze_label_noise.py outputs/pas_detection_*.json
```
