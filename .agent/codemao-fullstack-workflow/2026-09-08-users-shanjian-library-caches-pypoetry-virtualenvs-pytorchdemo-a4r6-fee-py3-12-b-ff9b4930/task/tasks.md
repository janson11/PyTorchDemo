# Transformer 示例修复任务清单

## 1. 输入计划和范围摘要

- 输入计划：[`.agent/codemao-fullstack-workflow/2026-09-08-users-shanjian-library-caches-pypoetry-virtualenvs-pytorchdemo-a4r6-fee-py3-12-b-ff9b4930/plan/architecture-plan.md`](../plan/architecture-plan.md)。
- 范围：修复手写 batch-first Transformer 的多头维度、完整层计算、词表投影、teacher-forcing 标签、验证损失、掩码设备和位置长度错误；隔离演示运行副作用；补足依赖声明和自动化回归测试。
- 非范围：不修改 `apply/oneTransformerDemo.py`，不替换为 `torch.nn.Transformer`，不引入数据集、分词、推理服务、持久化或生产部署能力。

## 2. 执行规则与全局约束

1. 保留 `Transformer(src, tgt)` 的构造函数参数顺序、batch-first 契约和直接执行脚本时的演示能力。
2. 维度不变量必须始终成立：`d_model % num_heads == 0`，`d_k == d_model // num_heads`，以及拆分前后元素数量守恒。
3. `src`、`tgt` 为 `(B, S)` 的 `LongTensor`；模型 logit 为 `(B, S_tgt, V)`；teacher forcing 的解码器输入是 `tgt[:, :-1]`，监督标签是 `tgt[:, 1:]`。
4. 掩码保持布尔值，并创建在输入相同设备上；无 CUDA 时 GPU 覆盖须显式跳过，不视为失败。
5. 保留用户已有的 `apply/TranformerModel.py` 工作树改动，只在计划范围内叠加修改；不得回退无关内容。
6. 测试不得启动默认的 100 轮训练，不得写入模型产物或访问网络服务。

## 3. 分阶段任务清单

### 阶段 A：共享注意力与编码器基础修复

进入条件：当前计划已确认，`apply/TranformerModel.py` 的用户改动仍保留。

- [ ] T001 修复 `apply/TranformerModel.py` 中 `MultiHeadAttention` 的每头维度和拆分/合并契约：令 `self.d_k = d_model // num_heads`，保留可整除校验，确认 Q/K/V 的 `(B,S,d_model)` 投影拆为 `(B,H,S,d_k)`、合并后恢复 `(B,S,d_model)`，并使缩放分母使用修正后的 `d_k`。
- [ ] T002 在 `apply/TranformerModel.py` 的 `EncoderLayer.forward` 补齐前馈网络、dropout、第二残差连接和 `norm2`；验证 `feed_forward` 与 `norm2` 均参与计算，且层输出仍为 `(B,S,d_model)`。

完成条件：默认 `d_model=512`、`num_heads=8` 的注意力拆分中单头宽度为 64，不再触发用户报告的 `view` 异常；编码器前馈参数可获得梯度。

### 阶段 B：完整模型、训练与验证路径修复

进入条件：T001、T002 完成，注意力与编码器层的形状契约稳定。

- [ ] T003 在 `apply/TranformerModel.py` 统一最终词表投影成员为语义明确的单一名称，并让 `Transformer.forward` 使用该成员，消除当前注册 `fc1` 却访问 `self.fc` 的属性错误；验证完整模型输出为 `(B,S_tgt,V)`。
- [ ] T004 在 `apply/TranformerModel.py` 修复 `generate_mask` 和 `PositionalEncoding` 的边界处理：no-peak 掩码在 `tgt.device` 上以布尔类型创建并正确广播；当输入长度超过 `max_seq_length` 时抛出包含实际长度与上限的清晰异常。
- [ ] T005 将 `apply/TranformerModel.py` 的训练和验证逻辑封装为可配置、可单步调用的辅助函数，并置于 `if __name__ == "__main__":` 保护下；训练与验证统一使用 `decoder_input=tgt[:, :-1]`、`target=tgt[:, 1:]` 和同一展平损失规则，保证一次 `backward()` 与 `optimizer.step()` 成功且模块导入没有训练副作用。

完成条件：小尺寸模型可完成前向、反向、优化器更新及 no-grad 验证损失；logit 行数与右移标签数量均为 `B * (S_tgt - 1)`；直接运行脚本仍可启动默认演示。

### 阶段 C：可复现环境与自动化验证

进入条件：T001 至 T005 完成，模块可被无副作用导入。

- [ ] T006 [P] 在 `tests/test_transformer_model.py` 建立小尺寸 pytest 回归覆盖：默认 512/8 的拆分合并、不可整除头数、self/cross attention 有限值、完整模型输出、编码器前馈梯度、单步训练、验证右移标签、CPU 掩码设备、位置编码上限与超限、以及导入无训练日志；CUDA 用例仅在可用时执行。
- [ ] T007 [P] 在 `pyproject.toml` 声明兼容 Python 3.12 的 PyTorch 运行依赖，并通过 Poetry 最小化更新 `poetry.lock`；检查解析结果与当前可验证环境一致，且不顺带升级无关依赖。

完成条件：测试和运行依赖均可由项目配置复现；默认示例的大模型训练不进入常规测试套件。

### 阶段 D：集成验收与交付准备

进入条件：T006、T007 完成。

- [ ] T008 使用 Poetry 环境执行 `python -m pytest -q`，并运行一个小配置的训练/验证 smoke test；核对 CPU 张量设备、有限损失、梯度和优化器更新，记录 CUDA 不可用时的跳过状态。
- [ ] T009 复查 `apply/TranformerModel.py`、`tests/test_transformer_model.py`、`pyproject.toml` 和 `poetry.lock` 的 diff：确认不修改 `apply/oneTransformerDemo.py`，没有导入副作用、数据迁移或外部状态写入；记录局部回滚点及最终验证命令。

完成条件：全部自动化测试通过，原始形状失败和其后的属性/验证形状错误均不可复现，diff 仅包含计划内文件与必要锁定依赖变化。

## 4. 依赖关系和关键路径

```text
T001 -> T002 -> T003 -> T004 -> T005 -> T006 -> T008 -> T009
                                   \-> T007 -/
```

关键路径为 `T001 -> T002 -> T003 -> T004 -> T005 -> T006 -> T008 -> T009`。`T007` 与测试代码编写修改不同文件，且不消费模型实现产物，可在阶段 C 与 T006 并行；二者均为 T008 的前置条件。

## 5. 可并行任务分组

| 分组 | 可并行任务 | 原因 | 汇合点 |
| --- | --- | --- | --- |
| C1 | T006、T007 | 分别修改测试文件与 Poetry 配置/锁文件；无共享代码或接口写入。 | T008 |

其余任务均在同一模型文件中修改，或消费前一任务的模型行为，因此严格串行执行。

## 6. 验收映射

| 计划验收场景 | 执行任务 | 可观察完成标准 |
| --- | --- | --- |
| 512 维、8 头的拆分合并 | T001、T006 | `(2,3,512) -> (2,8,3,64) -> (2,3,512)`。 |
| 非法头数的快速失败 | T001、T006 | `d_model=10, num_heads=3` 在构造期抛出明确异常。 |
| 三种注意力路径 | T001、T003、T006 | self/cross attention 输出维度正确且全为有限值。 |
| 完整模型输出投影 | T003、T006 | 输出 `(B,S_tgt,V)`，无 `AttributeError`。 |
| 编码器前馈已接入 | T002、T006 | `feed_forward.fc1.weight.grad` 非空且有限。 |
| teacher forcing 与训练更新 | T005、T006、T008 | 输出/标签展平行数一致，损失有限，反向和 step 成功。 |
| 验证标签与 no-grad | T005、T006、T008 | 验证使用 `target[:, 1:]`，不出现 64 对 6,336 的不匹配。 |
| 掩码设备 | T004、T006、T008 | CPU 通过；可用 CUDA 上掩码与输入同设备。 |
| 位置长度边界 | T004、T006 | 上限长度通过，超限产生清晰异常。 |
| 无导入副作用 | T005、T006 | 导入模块不打印 epoch、不训练。 |
| 依赖可复现 | T007、T008 | Poetry 配置和 lockfile 解析出 PyTorch，pytest 使用该环境通过。 |

## 7. 发布与回滚任务

没有线上发布、数据库迁移或模型文件迁移。T008 是发布前验证门槛，T009 是交付检查。

- 回滚 T001-T005：仅还原本次对 `apply/TranformerModel.py` 引入的局部修复，不覆盖用户预先存在的工作树内容。
- 回滚 T006：删除本次新增的测试文件即可，不影响运行时模型。
- 回滚 T007：同时还原 `pyproject.toml` 和对应 `poetry.lock` 片段，保持依赖元数据一致。
- 任一集成失败：停止在 T008，根据失败所属任务恢复到最近通过的任务边界，修复后重新执行 T006/T007 和 T008。

## 8. 总任务数和风险摘要

- 总任务数：9。
- 阶段分布：阶段 A 为 2 项，阶段 B 为 3 项，阶段 C 为 2 项，阶段 D 为 2 项。
- 可并行任务：2 项（T006、T007）。
- 阻断项：无。
- 最高风险任务：T001，因 `MultiHeadAttention` 被编码器自注意力、解码器自注意力和交叉注意力共同使用；通过精确形状测试和 T008 集成 smoke test 控制风险。
- 次高风险任务：T007，Poetry 锁文件可能引入转移依赖变更；通过最小化 diff 审查及与配置同步回滚控制风险。
