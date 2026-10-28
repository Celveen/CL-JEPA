# S7t3（CEM 规划器）：AdaJEPA、专家标签、v5a、v6a

- 流与 GD 版相同：每轮 7 个形状 × 50 个回合，三轮分别是测试种子 100、200、300，共 1050 个回合，逐回合配对。
- 规划器：CEM（200 个采样、取前 30、迭代 10 次，`conf/cont_plan_cem_pushobj.yaml`）。
- 四组：
  - **AdaJEPA**（w0）
  - **专家标签**（`cls3_oracle`）：按真实形状写入，从第一次规划起读本形状学习器；见过的形状也学。
  - **v5a**（`cls5a_free`）：写入与合并按 v5，熟悉的数据不调用学习器，小碎片吸收。
  - **v6a**（`cls6a_free`）：v6 的放宽写入和放宽合并，去掉形状分类器，熟悉的数据不调用学习器，小碎片吸收。
- 逐项统计见同目录 `rounds.md`。

## 1. 总体

| 分组 | AdaJEPA | 专家标签 | v5a | v6a |
|---|---:|---:|---:|---:|
| 全部（n=1050） | 72.3% | **78.6%** | 75.9% | 76.5% |
| 没见过·第 1 次（n=150） | 53.3% | 68.0% | 60.7% | 61.3% |
| 没见过·第 2 次（n=150） | 56.0% | 68.0% | 67.3% | 63.3% |
| 没见过·第 3 次（n=150） | 52.7% | 71.3% | 61.3% | 64.7% |
| 见过（三次合计，n=600） | 86.0% | 85.7% | 85.5% | 86.5% |

与 AdaJEPA 的配对差：专家标签 +6.3（139/73，p = 7×10⁻⁶）；v6a +4.2（87/43，p = 1.4×10⁻⁴）；v5a +3.6（90/52，p = 0.002）。
v6a − v5a = +0.6（63/57，p = 0.65）；v5a − 专家标签 = −2.7（p = 0.049）；v6a − 专家标签 = −2.1（p = 0.12）。

- **三组都显著好于 AdaJEPA，收益全在没见过的形状上，而且从第一次访问就开始。** GD 版第一次访问时各组和 AdaJEPA 差不多。
- **见过的形状四组都没有变化。** v5a、v6a 对熟悉的数据不调用学习器，没有伤到见过的形状。
- **v5a 和 v6a 没有差别。** 两者都比专家标签低 2–3 个点。

## 2. 按形状（第 1 / 2 / 3 次访问）

| 形状 | AdaJEPA | 专家标签 | v5a | v6a |
|---|---:|---:|---:|---:|
| I | 54 / 42 / 40 | 68 / 64 / 62 | 64 / 70 / 70 | 68 / 64 / 68 |
| square | 42 / 54 / 44 | **66 / 66 / 70** | 46 / 56 / 40 | 42 / 50 / 54 |
| small_tee | 64 / 72 / 74 | 70 / 74 / 82 | 72 / 76 / 74 | 74 / 76 / 72 |

- **I：** 不用标签的两组和专家标签一样好，甚至更好（比 AdaJEPA 高 20–30 个点）。
- **square：** 和专家标签的差距几乎全在这里。
  - 第 2、3 次访问、不熟悉的 square 回合（n=64）：专家标签 69%，v5a 45%，v6a 48%，AdaJEPA 47%。v5a 和 v6a 在其中 57–61 个回合读了学习器，但没有用。
  - 原因是学习器不够好。所读学习器在 square 上的 5 步误差比：专家标签 0.38 → 0.18（第 2 → 3 次）；v5a 0.71 → 0.42；v6a 0.51 → 0.47。
  - v5a 的 square 回合混在以 I 为主的大领域里（I 140、square 71）；v6a 的 square 领域混了 small_tee（square 76、small_tee 25）。
  - GD 版 v6 在 S7t3 上 square 涨了 18–20 个点，当时它读的学习器到第 3 次访问降到了 0.26。所以 square 能不能涨，跟着学习器在 square 上预测得准不准走。
  - 熟悉的 square 回合（n=36）：专家标签 67%，AdaJEPA 53%。v5a 和 v6a 在这些回合上按规则读原参数，这部分收益也拿不到。
- **small_tee：** 哪一组都没有稳定的收益。

## 3. 领域


### cls5a_free：4 个存活领域，共开过 9 个，待确认池 12 个；写入规则 {'pending': 30, 'new_from_pending': 9, 'direction': 29, 'learner': 213, 'none': 26, 'rejected': 27, 'familiar': 716}
| 领域 | 回合数 | 构成 | 流末 | 首次成熟 | 被读（按形状） | 创建于 |
|---|---:|---|---|---|---|---|
| d1 | 234 | I 140、square 71、small_tee 21、L 2 | 成熟 | I¹ | 212（I 113、square 75、small_tee 21、Z 1、L 1、+ 1） | I¹（new_from_pending） |
| d5 | 26 | small_tee 23、T 1、I 1、square 1 | 成熟 | small_tee¹ | 26（small_tee 17、I 4、T 2、square 2、Z 1） | small_tee¹（new_from_pending） |
| d6 | 5 | small_tee 4、L 1 | 未成熟 | — | 0（—） | small_tee¹（new_from_pending） |
| d7 | 4 | square 2、small_tee 2 | 未成熟 | — | 0（—） | square³（new_from_pending） |
合并/吸收：d0→d1（合并，I 14，I¹）；d2→d1（合并，I 6、square 2、small_tee 1，small_tee¹）；d3→d1（合并，square 7，square²）；d4→d1（合并，square 8、small_tee 4、L 1，square³）；d8→d5（吸收，small_tee 3，0.68，small_tee³）
I 各次访问的最终读取：第1次 None 22、d1 20、d0 7、d2 1；第2次 d1 45、d5 3、None 2；第3次 d1 48、d5 1、d4 1
square 各次访问的最终读取：第1次 None 25、d1 21、d2 3、d3 1；第2次 d1 24、None 23、d4 2、d5 1；第3次 d1 30、None 16、d4 3、d5 1
small_tee 各次访问的最终读取：第1次 None 43、d1 5、d2 2；第2次 None 24、d1 10、d5 8、d4 8；第3次 None 35、d5 9、d1 6

### cls6a_free：6 个存活领域，共开过 15 个，待确认池 12 个；写入规则 {'pending': 42, 'new_from_pending': 15, 'direction': 37, 'learner': 231, 'rejected': 13, 'familiar': 712}
| 领域 | 回合数 | 构成 | 流末 | 首次成熟 | 被读（按形状） | 创建于 |
|---|---:|---|---|---|---|---|
| d1 | 291 | I 146、square 93、small_tee 44、L 6、+ 1、Z 1 | 成熟 | I¹ | 155（I 125、square 17、small_tee 10、L 2、+ 1） | I¹（new_from_pending） |
| d8 | 7 | small_tee 7 | 未成熟 | small_tee³ | 1（small_tee 1） | small_tee¹（new_from_pending） |
| d10 | 6 | small_tee 5、Z 1 | 未成熟 | — | 0（—） | small_tee²（new_from_pending） |
| d11 | 3 | small_tee 2、+ 1 | 未成熟 | — | 0（—） | small_tee³（new_from_pending） |
| d12 | 3 | small_tee 3 | 未成熟 | — | 0（—） | small_tee³（new_from_pending） |
| d13 | 3 | small_tee 2、L 1 | 未成熟 | — | 0（—） | small_tee³（new_from_pending） |
合并/吸收：d2→d1（合并，I 5，I¹）；d0→d1（吸收，I 5，0.73，I¹）；d4→d1（合并，small_tee 4、L 1，I²）；d7→d3（吸收，small_tee 3，0.79，square²）；d6→d3（吸收，small_tee 2、L 1，0.80，small_tee²）；d5→d3（合并，small_tee 7、L 1，+²）；d3→d1（合并，square 76、small_tee 25、L 3、Z 1，square³）；d9→d1（吸收，square 3，0.74，square³）；d14→d1（吸收，small_tee 3，0.48，small_tee³）
I 各次访问的最终读取：第1次 d1 30、None 20；第2次 d1 47、None 3；第3次 d1 48、None 2
square 各次访问的最终读取：第1次 None 28、d3 19、d1 3；第2次 d3 28、None 21、d1 1；第3次 d3 23、None 14、d1 13
small_tee 各次访问的最终读取：第1次 None 46、d3 4；第2次 None 34、d3 10、d1 4、d5 2；第3次 None 43、d1 6、d8 1

## 4. 和 GD 版的对照（同一条流）

| | AdaJEPA | v5 / v5a | v6 / v6a |
|---|---:|---:|---:|
| GD（v5、v6） | 73.6% | 74.9%（+1.2，不显著） | 76.7%（+3.0，p = 0.013） |
| CEM（v5a、v6a） | 72.3% | 75.9%（+3.6，p = 0.002） | 76.5%（+4.2，p = 1.4×10⁻⁴） |

CEM 下持续学习的收益更早出现、也更稳定。
