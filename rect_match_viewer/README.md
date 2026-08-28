# Rect Match Viewer

用于单样本查看产品配置下的 MSK 矩形检测、机械 TXT 解析和匹配结果。

## 启动

在 `Detectors/rect_detector` 目录执行：

```bash
python rect_match_viewer/app.py --host 127.0.0.1 --port 7870
```

浏览器打开：<http://127.0.0.1:7870>

## 默认示例

页面默认填入：

- `configs/products/WFAW40UN.json`
- `temp/S26E31147-03/Light1-raw/IMAGE1_0113.msk`
- 自动匹配同名的 `IMAGE1_0113.txt`

路径为相对于工程根目录的路径，也可以填写绝对路径。

## 页面功能

- 显示 MSK 原图；
- 读取产品 JSON 中的 RAW 和矩形检测参数；
- 解析对应 TXT 的机械坐标；
- 执行矩形检测和机械坐标匹配；
- 显示匹配叠加图；
- 显示矩形数量、机械记录数量、匹配数量、RMSE 和最大残差；
- 显示每个检测框的坐标、`MX`、`MY` 和匹配残差；
- 未匹配的检测框会使用红色显示。

当前版本不执行 YOLO 推理，只负责 MSK 和 TXT 的矩形匹配调试。
