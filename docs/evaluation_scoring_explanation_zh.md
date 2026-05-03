# 图像编辑评测打分说明



## 1. 总体思路

当前评测分成两类：

- 语义级别评测：由多模态大模型对编辑结果进行主观判分
- 像素级别评测：将预测图与参考图做客观图像相似度计算

最终每个样本会得到两组分数：

- 语义分：`IF`、`TA`、`VC`、`LP`，以及新增的 `LSF`
- 像素分：`MSE`、`PSNR`、`SSIM`、`LPIPS`

## 2. 语义级别评测怎么打分

语义评测的核心实现是 [evaluate_semantic.py](../evaluation/llm_evaluation/evaluate_semantic.py)。

### 2.1 每个样本实际用了哪些文件

对于一个样本 `category / id / lang`，脚本会读取：

- 元数据 JSON：`{Category}_{ID}.json`
- 输入图：`1.jpg`
- 参考输出图：由 JSON 中的 `output_image` 字段指定
- 模型预测图：`{ID}_edited.png`

也就是说，语义评分不是只看预测图，而是把下面三张图一起交给评测模型：

- 输入图 `input_image`
- 预测图 `pred_image`
- 参考图 `output_image`

同时还会传入：

- `prompt`
- `editing_method`

### 2.2 语义评测的执行方式

对每个样本，脚本会把图像转为 base64，然后通过 OpenAI Responses API 调用多模态模型。

不是一次调用给出所有分数，而是“每个维度单独调用一次”。因此一个样本通常会发生：

- `IF` 一次
- `TA` 一次
- `VC` 一次
- `LP` 一次
- 如果启用 `LSF`，还会多出一套“两段式判断”：
  - `LSF_TRACE` 一次：先定位目标编辑文本并抄写
  - `LSF` 一次：再只对目标文本的语言/书写系统忠实度打分

注意：

- 对 `text_delete` 样本，`LSF` 会标记为 `not_applicable`
- 因此非删除样本的 LSF 成本通常比其他单维度更高

因此，当前语义评测本质上是一个 **LLM-as-a-judge** 流程。

### 2.3 语义评测维度

#### 2.3.1 IF：Instruction Following，指令遵循

关注点：

- 模型是否执行了正确的编辑类型
- 是否只执行了要求的那一种操作
- 是否出现额外、不必要的改动

评分区间：`0-5`

- `5`：完全按照要求执行了指定操作，没有额外修改
- `4`：操作正确，但有极轻微额外变化
- `3`：总体方向正确，但有比较明显的多余修改或遗漏
- `2`：做了对应操作，但偏差较大
- `1`：操作类型错误，或执行质量很差
- `0`：基本没有按要求完成

#### 2.3.2 TA：Text Accuracy，文本准确性

关注点：

- 文本内容是否改对
- 拼写、大小写、翻译是否正确
- 最终文本是否与目标输出一致

评分区间：`0-5`

- `5`：目标文本完全正确
- `4`：只有轻微拼写或格式差异
- `3`：主体正确，但仍有明显错误
- `2`：多个文本元素不正确
- `1`：大部分错误，或本应修改但没有改
- `0`：文本结果完全错误

#### 2.3.3 VC：Visual Consistency，视觉一致性

关注点：

- 新文字是否与原图风格融合
- 字体、颜色、边缘、透视、对齐是否自然
- 是否出现粘贴感、光晕、锯齿等伪影

评分区间：`0-5`

- `5`：视觉融合非常自然
- `4`：整体很好，只有很小瑕疵
- `3`：存在可见但不算严重的违和感
- `2`：明显不协调
- `1`：看起来像后贴上去的
- `0`：严重不自然或不可读

#### 2.3.4 LP：Layout Preservation，布局保留

关注点：

- 非目标区域是否保持不变
- 背景和构图有没有被无关改动破坏

评分区间：`0-5`

- `5`：除目标区域外基本完全不变
- `4`：几乎不变，只有极小扰动
- `3`：有轻微无关变化
- `2`：有较多无关变化
- `1`：大量非目标区域被改动
- `0`：整体布局都被破坏

#### 2.3.5 LSF：Language / Script Fidelity，语言/书写系统忠实度

这个维度是后来新增的，用来专门评估“目标编辑文本”的书写系统是否正确。

它关注的不是“这句话大意对不对”，而是：

- 字符是否写对
- 是否有漏字、多字、错字
- 变音符、重音、声调、附标是否丢失或错误
- `RTL / LTR` 顺序是否正确
- 标点、括号方向是否正确
- 是否出现 script mixing
- 语言特有排版或字形 shaping 是否异常

评分区间同样是 `0-5`：

- `5`：目标文本的书写系统完全正确
- `4`：只有一个轻微书写系统问题
- `3`：有明确字符级或附标级错误
- `2`：有多处明显错误，或方向 / script 问题较重
- `1`：大部分目标文本书写系统错误明显
- `0`：目标文本缺失、不可读，或明显是错误 script / 错误方向

LSF 采用两段式流程：

1. `LSF_TRACE`
   比较 `input_image` 和 `output_image`，先确定“哪一段文字才是被编辑目标”，并抄写：
   - `input_text_before`
   - `output_text_expected`
   - `pred_text_observed`

2. `LSF`
   只基于 trace 里识别出的目标文本，判断书写系统忠实度。

这意味着 LSF **不是整图 OCR**，也不是对整张图所有文字统一打分，而是只评目标编辑文本。

#### 2.3.6 LSF 和 TA 的关系

`TA` 和 `LSF` 看起来都在评“文本对不对”，但它们解决的是不同层次的问题：

- `TA` 关注的是 **文本内容语义是否正确**
  - 词有没有改对
  - 拼写是否大体正确
  - 翻译、大小写、目标内容是否匹配

- `LSF` 关注的是 **书写系统层面的忠实度**
  - 字符级错误
  - 变音符 / 声调 / 重音
  - `RTL / LTR`
  - script mixing
  - 语言特有排版

可以把它们理解成：

- `TA` 更偏“你写的是不是那句话”
- `LSF` 更偏“你是不是用正确的字形和书写系统把它写出来了”

这两个维度可能一致，也可能不一致。

例如：

- 越南语目标文本本来有完整声调，但预测图把重音全部丢了  
  这时 `TA` 可能仍然偏高，因为主体词还能认出来；但 `LSF` 应该明显下降。

- 希伯来语或阿拉伯语的词本身对了，但方向错了、附标错了，或者括号方向不对  
  这时 `TA` 也可能不算太低，但 `LSF` 应该扣分。

- 反过来，如果字符和书写系统都正确，只是改错了目标词本身，那么 `TA` 会低，但 `LSF` 未必低。

因此，当前实现里：

- `TA` 和 `LSF` 会同时保留
- 它们不会被合并成一个“文本总分”
- 分析时应该把 `LSF` 视为 `TA` 的补充诊断维度，而不是它的替代项

### 2.4 语义分数的输出格式

单条样本的结果结构大致如下：

```json
{
  "task_id": "TextEditing_Quotes_001_en",
  "category": "Quotes",
  "model": "gpt-image-1.5",
  "id_str": "001",
  "lang": "en",
  "operation": "color_change",
  "evaluation_results": {
    "IF": {"IF": 5, "rationale": "..."},
    "TA": {"TA": 4, "rationale": "..."},
    "VC": {"VC": 5, "rationale": "..."},
    "LP": {"LP": 5, "rationale": "..."},
    "LSF": {
      "LSF_status": "scored",
      "LSF": 4,
      "error_tags": ["accent_error"],
      "rationale": "..."
    }
  },
  "evaluation_traces": {
    "LSF_TRACE": {
      "trace_status": "success",
      "overall_trace_confidence": 0.92,
      "target_segments": [
        {
          "segment_id": "seg_1",
          "output_text_expected": "café",
          "pred_text_observed": "cafe",
          "pred_found": true
        }
      ]
    }
  },
  "schema_version": "2.0"
}
```

也就是说，语义评测会同时输出：

- 该维度的整数分数
- 该维度的文字解释 `rationale`
- 对于 `LSF`，还会额外保存 `LSF_TRACE`
- `LSF` 不一定总是有整数分数，也可能出现：
  - `LSF_status = "not_applicable"`：例如 `text_delete`
  - `LSF_status = "unscorable"`：trace 不够可靠
  - `LSF_status = "error"`：请求或解析失败

### 2.5 语义分数怎么汇总

语义分数的汇总方式非常直接：**对同一维度做算术平均**。

脚本会分别计算：

- 整体平均分
- 按编辑类型 `operation` 分组的平均分
- 按语言 `lang` 分组的平均分

例如：

- 所有样本的 `IF` 平均值
- 所有 `color_change` 样本的 `TA` 平均值
- 所有 `zh` 样本的 `VC` 平均值
- 所有有效样本的 `LSF` 平均值

对于 `LSF`，当前实现还会额外统计：

- `LSF_valid_count`
- `LSF_not_applicable_count`
- `LSF_unscorable_count`

当前实现里，语义脚本本身不会把 `IF/TA/VC/LP/LSF` 合成一个总分，而是保留各维度平均值，由后续分析脚本统一合表。

## 3. 像素级别评测怎么打分

像素级评测分为两组指标：

- `MSE / PSNR`
- `SSIM / LPIPS`

它们的共同特点是：

- 不使用 `prompt`
- 不调用大模型
- 使用原图 `input_image`
- 使用 mask
- 比较的是“预测图”和“原图在非编辑区域的差异”

因此，像素级评测的本质是：

```text
原图  vs  预测编辑图
只在 mask 外背景区域计分
```

### 3.1 每个样本实际用了哪些文件

对一个样本，像素评测只需要：

- 元数据 JSON：用来找到 `input_image`、`output_image` 和 `editing_method`
- 输入图：通常是 `1.jpg`
- 模型预测图：`{ID}_edited.png`
- 输入图对应 mask：例如 `1_mask.jpg`
- 输出图对应 mask：例如 `text_color_change_1_mask.jpg`

元数据在这里的作用是：

- 确定输入图文件名
- 确定输出图文件名，从而推导目标编辑区域的 mask 文件名
- 记录该样本属于哪种 `editing_method`，便于后续分组统计

实现入口是：

- [evaluate_mse_psnr_masked.py](../evaluation/pixel_evaluation/evaluate_mse_psnr_masked.py)
- [evaluate_ssim_lpips_masked.py](../evaluation/pixel_evaluation/evaluate_ssim_lpips_masked.py)

### 3.2 mask 是怎么用的

脚本会读取两张 mask：

- 输入图 mask：标记原始文字区域
- 输出图 mask：标记编辑后目标区域

这两张 mask 会先做 OR 合并，再取反，得到“背景区域 mask”。

最终所有像素指标都只在这个 **mask 外背景区域** 上计算。

## 4. MSE / PSNR 是怎么计算的

实现文件是 [evaluate_mse_psnr_masked.py](../evaluation/pixel_evaluation/evaluate_mse_psnr_masked.py)。

### 4.1 预处理方式

`MSE / PSNR` 在正式计算前，会先做：

1. 读取原图、预测图、两张 mask
2. 合并两张 mask
3. 使用 `SIFT + FLANN + affine transform` 把预测图对齐到原图坐标系
4. 如果对齐后图像尺寸与原图不同，则 resize 到原图尺寸
5. 如果 mask 尺寸与原图不同，也会 resize 到原图尺寸

因此这套 `MSE / PSNR` 是一套 **容错型 masked background evaluation**。

### 4.2 MSE

代码中的计算方式是：

```python
masked_squared_error = squared_error * inv_mask[..., None]
mse = np.sum(masked_squared_error) / total_data_points
```

含义：

- 只对背景区域像素计算平方误差
- 再按背景区域总像素数做平均

解释：

- `MSE` 越小越好
- `MSE = 0` 表示在背景区域内，预测图与原图完全一致

### 4.3 PSNR

代码中的计算方式是：

```python
psnr = 10 * np.log10(255.0 ** 2 / mse)
```

特殊情况：

- 如果 `mse == 0`，则 `psnr = inf`

解释：

- `PSNR` 越大越好
- 它本质上是根据 `MSE` 推导出的对数质量指标

### 4.4 并行方式

当前这版 `MSE / PSNR` 支持：

- `--workers 1`：串行
- `--workers > 1`：使用线程池并行处理多个样本

默认会启用一个适中的线程数。

### 4.5 MSE / PSNR 的汇总方式

单个样本会输出：

- `task_id`
- `id_str`
- `lang`
- `operation`
- `mse`
- `psnr`
- `input_image`
- `pred_image`
- `mask_images`

之后脚本会做三层平均：

- 全部样本的平均 `mse` 与平均 `psnr`
- 每种 `operation` 的平均 `mse` 与平均 `psnr`
- 每种 `lang` 的平均 `mse` 与平均 `psnr`

## 5. SSIM / LPIPS 是怎么计算的

实现文件是 [evaluate_ssim_lpips_masked.py](../evaluation/pixel_evaluation/evaluate_ssim_lpips_masked.py)。

### 5.1 预处理方式

`SSIM / LPIPS` 的预处理现在是：

1. 读取原图与预测图
2. 读取两张 mask
3. 如果预测图尺寸与原图不同，则先 resize 到原图尺寸
4. 如果 mask 尺寸与原图不同，也会先 resize 到原图尺寸
5. 合并两张 mask，再取反为背景区域 mask
6. 图像归一化到 `[0, 1]`
7. 应用背景 mask
8. 转成 tensor 并送入 `cuda` 或 `cpu`

也就是说，当前 `SSIM / LPIPS` 已经不是“尺寸不同就失败”，而是会先统一 resize 到原图尺寸再打分。

### 5.2 SSIM

脚本调用的是 TorchMetrics 中的：

```python
StructuralSimilarityIndexMeasure(data_range=1.0)
```

计算方式是：

```python
ssim = self.ssim_metric(pred_tensor, ref_tensor).item()
```

解释：

- `SSIM` 越高越好
- 通常范围在 `[0, 1]`
- 它关注的是结构相似性，而不只是逐像素误差

### 5.3 LPIPS

脚本调用的是：

```python
LearnedPerceptualImagePatchSimilarity(net_type='squeeze')
```

调用前，图像会从 `[0, 1]` 映射到 `[-1, 1]`：

```python
lpips = self.lpips_metric(pred_tensor * 2 - 1, ref_tensor * 2 - 1).item()
```

解释：

- `LPIPS` 越低越好
- 它使用深度特征衡量感知差异，不是简单的像素差

### 5.4 SSIM / LPIPS 的汇总方式

与 `MSE / PSNR` 相同，脚本会输出每个样本的：

- `ssim`
- `lpips`
- `operation`
- `lang`

然后分别计算：

- 全体样本平均 `ssim`、平均 `lpips`
- 每种 `operation` 的平均 `ssim`、平均 `lpips`
- 每种 `lang` 的平均 `ssim`、平均 `lpips`

## 6. 最终分析时这些分数怎么合并

语义和像素评测的原始结果会被后续分析脚本合并到统一表格里。合并后的样本级指标通常包括：

- `IF`
- `TA`
- `VC`
- `LP`
- `SE`（如果有）
- `mse`
- `psnr`
- `ssim`
- `lpips`

之后再按：

- 类别
- 编辑类型
- 语言

做平均统计和可视化。

## 7. 当前实现要特别注意的几点

### 7.1 当前像素打分使用 mask，且只看背景区域

当前像素评测会同时使用：

- `1_mask.jpg`
- `text_*_mask.jpg`

因此现在的 `MSE/PSNR/SSIM/LPIPS` 都不是整图比较，而是 **mask 外背景区域比较**。

### 7.2 当前像素打分是“原图对比预测图”，不是“参考输出图对比预测图”

像素评测现在不再直接比较：

```text
预测图 vs 参考输出图
```

而是比较：

```text
原图 vs 预测图
```

并通过 mask 把目标编辑区域排除掉。

### 7.3 `LSF` 是新增诊断维度，不等于 `TA`

`LSF` 当前已经接入语义评测输出结构，并且和 `IF / TA / VC / LP` 一样进入同一条结果记录。

但在解释上，应把它看成“文字内容评测的补充维度”，而不是简单把 `TA` 再拆细一次：

- `TA` 回答“改成的文本内容是不是对的”
- `LSF` 回答“这段目标文本是不是被正确地用对应语言 / 书写系统写出来了”

因此，多语种分析时，尤其是：

- 阿拉伯语
- 希伯来语
- 越南语
- 孟加拉语
- 以及带大量重音或附标的语言

通常应该同时看 `TA` 和 `LSF`，不能只看 `TA`。

### 7.4 `change_font` 在语义提示词中的覆盖不完整

数据集中存在 `change_font` 类型，但 `IF` 的提示词里列举的操作类型没有显式包含它。

代码仍然会把真实 `editing_method` 传给评测模型，所以流程可以运行；但从提示词设计上看，`change_font` 的评分规范没有其他几类那么明确。

## 8. 一句话总结

当前这套打分机制可以概括为：

- 语义分：让多模态大模型看“输入图 + 预测图 + 参考图 + 指令”，按 `IF/TA/VC/LP/LSF` 做 0 到 5 分判分
- 像素分：直接把预测图和参考图做整图相似度计算，得到 `MSE/PSNR/SSIM/LPIPS`

也就是说：

- 语义分是 **LLM judge**
- 像素分是 **客观图像指标**
- 当前实现 **不使用 mask**
