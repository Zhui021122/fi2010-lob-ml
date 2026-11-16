# FI-2010 限价订单簿机器学习项目

这是面向金融机器学习工程岗位的完整作品：使用公开 FI-2010 限价订单簿数据预测短期中间价方向，并提供数据校验、只使用当前及历史信息的特征、按交易日划分的训练/验证/测试、模型文件、批量预测、测试和 CI。

在固定数据版本的第 8–10 个交易日留出集上，宏平均 F1 为 **0.815**，平衡准确率为 **0.805**（101,269 条样本）。第 7 日只用于选模型；以上是分类指标，**不是策略收益**。完整结果见 [RESULTS.md](artifacts/RESULTS.md)，局限性见 [LIMITATIONS.md](docs/LIMITATIONS.md)。

## 运行

安装 Python 3.10+，在仓库根目录执行：

```bash
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m lobml validate --data examples/fi2010_sample.csv
python -m lobml train --data examples/fi2010_sample.csv --out demo_artifacts
python -m pytest -q
```

仓库自带 5,000 行真实数据样本，便于无网络运行完整流程；样本实验指标不等于上方全量指标。要复现全量结果，继续执行：

```bash
python -m lobml download
python -m lobml validate --data data/fi2010.csv
python -m lobml train --data data/fi2010.csv --out artifacts
```

`download` 会下载固定版本的公开整理版 CSV（约 59 MB）并验证 SHA-256；直接下载失败时会尝试 Git。大文件不进入 Git 仓库。若已自行下载，可用 `--data 路径`。批量预测：

```bash
python -m lobml predict --data data/fi2010.csv --model artifacts/model.joblib --output predictions.csv
```

输入需要 `STOCK`、`DAY`、40 个价格/数量列；同一股票与交易日的记录必须连续且按事件顺序排列。可不提供标签。数据来源和许可见 [DATA.md](docs/DATA.md)。

## 求职展示重点

可以重点讲清楚为什么随机打散会泄漏未来信息、如何通过分日切分与日末剔除控制标签窗口、为什么只用验证日选模型，以及为何分类 F1 不能直接转化为可交易收益。
