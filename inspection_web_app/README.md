# 工业检测图片评价台

这是一个不依赖 Gradio 的本地网页应用，保留原脚本的图片筛选、分页显示和自动裁边缓存，并新增每张图片的评价能力：

- 根据图片文件名中的 `OK` / `NG` 自动显示“模型判定为”：`合格品` / `缺陷品`
- 顶部可筛选模型判定为 `合格品` 或 `缺陷品`
- 人工对分类结果打标为 `分类正确` 或 `分类错误`
- 保存评价前必须先完成“对分类结果做人工打标”
- 已保存或导入后已有人工打标的图片卡片会变色，表示已打标
- 当人工打标为 `分类错误` 时，可选择 `有缺陷但框是绿色的`
- `有缺陷但框是绿色的` 支持绿色画框，并独立保存
- 独立选择目标检测问题：`漏检`、`错检`，它们不会自动改变整图分类结果
- 选择 `漏检` 或 `错检` 后，都可在图片上拖拽画框
- `漏检` 框和 `错检` 框使用不同颜色，并分别保存
- 评价结果保存到 `annotations/annotations.json`
- 页面右上角可以下载当前所有评价 JSON
- 页面右上角可以导入别人导出的 JSON，并在同一图片路径下回显评价结果
- `Shuffle` 默认开启，会先打乱完整图片列表，再按页显示
- `JSON优先` 默认开启，导入或已保存评价的图片会排在未评价图片前面
- `查找图片` 支持按文件名、完整路径、JSON 中的 `imageName`、备注搜索
- 右侧显示基于所有已保存评价的混淆矩阵、错检率和漏检率
- 右侧同时按 `light_1`、`light_2` 等光照通道分别显示混淆矩阵和错检/漏检率
- 当筛选为某一类模型判定时，底部可将当前页剩余未打标图片批量默认标注为 `分类正确`

## 启动

在本目录运行：

```powershell
& "C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\app.py --host 127.0.0.1 --port 7868
```

然后打开：

```text
http://127.0.0.1:7868
```

如果你已经启动过旧版本，需要先关闭原来的 PowerShell 服务窗口，重新执行启动命令，页面才会加载最新功能。

## 导入与导出

点击右上角“下载评价 JSON”可以得到当前所有评价。点击“导入 JSON”选择别人导出的文件，系统会把其中的评价合并到本地 `annotations/annotations.json`。

导入后，页面依靠 `originalPath` 匹配图片。因此对方导出的 JSON 里图片路径需要和你当前机器上的图片路径一致，才能在加载图片时自动回显。
如果路径不一致，但图片文件名一致，页面会尽量用 JSON 里的 `imageName` 和当前图片文件名匹配。

新的评价字段含义：

```json
{
  "originalPath": "C:\\data\\light_1\\yolo_pred_images\\sample.jpg",
  "verdict": "分类错误",
  "greenDefect": true,
  "greenDefectRegions": [{"x": 30, "y": 40, "w": 60, "h": 70}],
  "detectionIssues": ["漏检", "错检"],
  "missRegions": [{"x": 10, "y": 20, "w": 100, "h": 80}],
  "falseRegions": [{"x": 200, "y": 50, "w": 120, "h": 90}],
  "note": "人工判定分类错误，并存在绿色缺陷框和目标检测问题"
}
```

其中 `verdict` 表示“对分类结果做人工打标”的结果，`greenDefect` 表示是否选择“有缺陷但框是绿色的”，`detectionIssues` 表示目标检测问题。`greenDefectRegions` 只保存绿色缺陷框，`missRegions` 只保存漏检框，`falseRegions` 只保存错检框。

## 混淆矩阵统计

右侧混淆矩阵只统计已经保存评价、且能从文件名解析出模型判定的记录。模型判定为 `缺陷品` 视为预测正样本，模型判定为 `合格品` 视为预测负样本。

- TN：模型判定为合格品，人工打标为分类正确
- FN：模型判定为合格品，人工打标为分类错误
- FP：模型判定为缺陷品，人工打标为分类错误
- TP：模型判定为缺陷品，人工打标为分类正确

错检率 = `FP / (FP + TN)`。

漏检率 = `FN / (FN + TP)`。

总矩阵下面会按图片路径或文件名中的 `light_数字` 分组，分别计算每个 light 的 TP、FN、FP、TN、错检率和漏检率。

## 数据目录要求

目录结构沿用原 Gradio 代码：

```text
Base Directory/
  light_1/yolo_pred_images/*.jpg
  light_2/yolo_pred_images/*.jpg
  light_3/yolo_pred_images/*.jpg
  light_4/yolo_pred_images/*.jpg
```

如果根目录不在应用目录下，请在页面的“数据根目录”中填写绝对路径。
