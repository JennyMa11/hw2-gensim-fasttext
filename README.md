# Gensim 表示学习与官方 fastText 分类

仓库地址：[JennyMa11/hw2-gensim-fasttext](https://github.com/JennyMa11/hw2-gensim-fasttext)

[完整中文报告（在线阅读）](report.md) · [PDF 报告](report.pdf) · [逐篇预测](results/predictions.csv) · [复现说明](#环境与复现)

## 主要结果（分类 seed=42，固定 25 轮）

| 类别数 | 从零训练：准确率 / 宏 F1 | FastText 初始化：准确率 / 宏 F1 |
| --- | --- | --- |
| 2 | 0.833 / 0.829 | 1.000 / 1.000 |
| 4 | 0.521 / 0.490 | 0.833 / 0.834 |
| 8 | 0.281 / 0.242 | 0.792 / 0.790 |

**解释范围：**标签由标题关键词生成；正文不含编号为 001 的标题行。追加的 100/200 轮诊断显示，从零训练增加轮次后会追上并超过预训练模型，因此不能把 25 轮对照解释为最终精度优势。详见完整报告。

![分类准确率与宏 F1](figures/classification_scores.png)


本作业包含独立编写、实际执行的完整实验。先读 `report.pdf` 或 `report.md`；所有数值可追溯到 `results/`，没有沿用参考作业的实验分数。

## 实验内容

- 在同一份人民日报预训练语料上训练 skip-gram Word2Vec 和 FastText。
- 检查词近邻、OOV、固定词组的余弦相似度，并复算 FastText 子词组合公式。
- 8 类各 60 篇弱标签文章，每类训练 48 篇、测试 12 篇；构建嵌套的 2／4／8 类任务。
- 官方 fastText 有／无预训练向量对照，分类随机种子为 42、43、44。
- 扩展：8 类 Word2Vec 初始化、每类 12／24／48 篇的嵌套训练规模曲线。
- 事后诊断：25／100／200 轮训练敏感性，揭示从零训练在 25 轮时的收敛不足；不更改主表或按测试集选择超参数。
- 分类测试文章在预训练前整篇剔除，分类正文不含标题。

**设置说明：**这是以 PeopleDaily1998 为语料的独立实验方案，并非题目示例 THUCNews 实验的原样复现。主题标签来自标题规则，未声称是逐篇人工金标准。结果只能评价所选弱标签分布。

## 环境与复现

实测 Python 3.12.14 / macOS arm64。建议 Python 3.12；依赖锁定在 `requirements-lock.txt`。`fasttext-wheel` 提供导入名为 `fasttext` 的官方算法实现；如果平台没有预编译 wheel，pip 会从源码构建，需要 C++ 编译工具。锁定 NumPy 1.26.4 避免 fastText 旧接口的 NumPy 2 兼容问题。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
bash run_all.sh
```

运行脚本自动下载固定 commit 的 19.3 MiB 原始 ZIP，并校验 SHA-256。完整流程通常需数分钟到十几分钟，取决于机器与网络。模型及导出向量会占用数百 MiB。首次下载或首次导入科学计算库可能更慢。

分类模型训练后，可输入**已分词正文**预测：

```bash
python scripts/07_predict.py --text '中国队 足球 比赛 球员 训练 冠军'
```

## 文件

| 路径 | 内容 |
| --- | --- |
| `report.pdf` / `report.md` | 完整中文报告与图表 |
| `scripts/common.py` | 类别、关键词、预先指定词组与超参数 |
| `scripts/01_prepare.py` | 下载校验、解析、去重、弱标签和文章划分 |
| `scripts/02_embeddings.py` | 实际训练两个 Gensim 模型，导出 `.vec` |
| `scripts/03_embedding_analysis.py` | 近邻、OOV、余弦与组合公式核对 |
| `scripts/04_classify.py` | 分类对照、三个种子、扩展、逐篇预测 |
| `scripts/05_verify_and_plot.py` | 防泄漏与结果核验、绘图 |
| `scripts/06_report.py` | 根据结果生成报告 |
| `scripts/07_predict.py` | 调用已训练分类模型 |
| `scripts/08_epoch_diagnostic.py` | 事后轮次诊断与逐篇预测核验 |
| `scripts/09_package.py` | 生成轻量提交包及 SHA-256 文件清单 |
| `results/sample_manifest.csv` | 480 篇的 ID、类别、划分、标题、命中词、正文哈希 |
| `results/predictions.csv` | 各次运行逐篇预测与概率 |
| `results/classification_runs.json` | 单次指标、逐类报告、混淆矩阵、参数和训练时间 |
| `results/classification_summary.json` | 三种子均值与样本标准差 |
| `results/verification.json` | 实验数据与指标一致性核验 |
| `results/environment.json` | 实际运行环境 |
| `figures/` | 四张中文结果图 |
| `data/selection.json`、`data/train_*.txt`、`data/test_*.txt` | 固定划分及官方 fastText 输入 |
| `models/`（重建或本地完整目录） | Gensim 模型、向量、8 类分类器 |

轻量提交 ZIP 不包含原始全文、`articles.jsonl`、预训练 `sentences.txt` 或大型模型；这些可由源码重建。本地完整目录保留本次生成的模型和数据。图表使用系统中文字体；Linux 建议安装 Noto Sans CJK，若缺字体不会影响训练，但图中的汉字可能无法显示。交付 PDF 已嵌入中文字体。重新生成 PDF 时，macOS 自动使用宋体；其他平台可设 `HW2_PDF_FONT` 指向含中文的 TrueType 字体文件，否则使用 ReportLab CID 字体（阅读器需支持中文映射）。

## 数据来源与校验

- 镜像：https://github.com/chenhui-bupt/PeopleDaily1998
- 固定 commit：`4020ecc558c554c6f3c9bd81a606d2197e28e628`
- 原始 ZIP：`https://raw.githubusercontent.com/chenhui-bupt/PeopleDaily1998/4020ecc558c554c6f3c9bd81a606d2197e28e628/199801.zip`
- SHA-256：`17474bbf2b360921f2eae0089fd83e22601e5a0649943c3f59403650997b4ed2`
- ZIP 名称虽为 199801，实际包含 199801-199806 六个文本文件。数据原始许可与用途说明见源仓库及压缩包中的声明。

## 阅读结果的限制

标题关键词提供弱监督而非主题金标签。排除多主题关键词命中不能消除标签噪声；正文截断和标题／正文的主题差异也会影响结果。三种子仅改变分类器随机种子，固定预训练与文章划分，标准差不包含全流程的不确定性。学习曲线保持无标签语料固定，不能当作预训练语料规模 scaling 的结论。
