# StepContract

为你声明的 PyTorch 训练意图做检查：哪些参数必须参与训练、梯度应累加几次、清梯度与更新步骤是否符合预期。

```sh
python -m pip install .
stepcontract demo --case healthy --html healthy.html
stepcontract demo --case missing-optimizer --html missing.html
stepcontract demo --case early-zero --html early-zero.html
```

后两条是实际执行的错误演示，退出码 1 是预期结果。报告显示参数归属和真实事件，不含原始权重、梯度或数据。

通过 `StepContract(model, optimizer, required_parameters=[...])` 明确意图，再显式调用审计对象的 `zero_grad/backward/step`。工具不自行调整学习率或冻结策略，不使用全局 hook。正常 SGD/AdamW 已与不加审计器的控制组比较。

首版仅支持 CPU、稠密连续 FP32、单进程 eager；不认证 AMP、分布式或编译图。零梯度、零学习率和合法冻结不会因为“没有更新”就自动判错。O(P) 精确快照有开销，预算不是全进程内存上限。

事件证据分“准入前拒绝”与“已准入调用”：前者（如非布尔 `set_to_none`、事件历史上限、预检发现不支持的优化器模式）抛错但不开事件、不消耗历史，也不会追溯否定已完成步骤；此前的 PASS 不认证这次被拒绝的调用。已准入事件记录 `operation_started`/`operation_returned` 与 `outcome`；每条 finding 都带真实 `event_index` 或 `null` 以及 `origin`（construction/event/report）。首个已准入调用异常后，该审计对象进入粘性 ERROR，之后重试成功也不能恢复 PASS。快照失败时优化器不会被调用，已完成步数不变。

完整 API 与限制见[英文文档](README.md)。本工具检查声明的局部契约，不证明全局训练正确或模型质量。MIT，AI 辅助实现，尚无 PyPI 发行或真实用户采用声明。

## 快速开始与支持状态

### 问题
显式声明训练步骤意图，防止有梯度的参数静默遗漏优化器，或违反声明的步骤顺序。

### 安装
```sh
python -m pip install .
```

### 成功命令
```sh
stepcontract demo --case healthy --html healthy.html
```
预期：健康 CPU 训练窗口以 PASS 完成；退出码 0。

### 故意失败命令
```sh
stepcontract demo --case missing-optimizer --html missing.html
```
预期：缺失优化器对照报告契约违规；退出码 1。

### 证据与覆盖边界
证据使用真实 SGD 与 AdamW 对照、相同更新检查、合法冻结与累积、缺失优化器参数、提前/外部清梯度、非有限梯度、身份变化和 CLI 报告。覆盖范围限于通过审计对象观察到的调用；不认证任意外部操作或整个程序的正确性。

### 支持状态
版本 0.1.0 支持 Python 3.10+、CPU PyTorch，以及限定范围内的 eager FP32 SGD/AdamW 工作流。文档路径是源码检出；不声称已发布到 PyPI。维护为尽力而为，不提供 SLA。

### 限制
分布式训练、AMP、编译图、优化器 closure、可微分/可捕获/融合模式，以及共享存储的独立 Parameter 对象不在范围内。快照开销为 O(P)，名称和形状可能敏感。
