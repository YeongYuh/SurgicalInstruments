# Demo 手術器械套件

已完成設定，可直接在畫面右上角選用。

| 項目 | 內容 |
|---|---|
| 模型 | `models/demo/best.onnx`（YOLOv9 GELAN-c segmentation，17 類別） |
| adapter | `yolov9_seg_onnx` |
| 推理尺寸 | 640×640 |
| 每類單重 | `class_weight.json`（17 種器械，公克） |
| 手術類型 | `surgery_instruments.json`（SurgeryA–D） |
| 預設標準數量 | `standards.json` 全部為 0 —— 請先在畫面上選擇手術類型 |

## 模型來源與匯出

原始權重是 **WongKinYiu/yolov9** 的 `gelan-c-seg` 訓練結果
（`runs/train-seg/exp11/weights/best.pt`，17 類別手術器械）。
它**不是** Ultralytics 匯出，`ultralytics.YOLO()` 無法載入
（checkpoint 內含 `models.common.RepNCSPELAN4`、`models.yolo.Segment`）。

`models/demo/best.onnx` 由官方程式碼匯出：

```bash
git clone --depth 1 https://github.com/WongKinYiu/yolov9.git
cd yolov9
python export.py --weights <best.pt> --include onnx \
                 --imgsz 640 640 --batch-size 1 --opset 12 --device cpu
```

> Jetson 上的 yolov9 `utils` 會 import `IPython`（只用於 notebook 偵測）。
> 沒有安裝時建立一個空的 `IPython` / `IPython.display` stub 即可，
> 不需要把 IPython 裝進正式 venv。

匯出的圖包含 `names` metadata，adapter 直接從模型讀類別名稱，
不需要在 manifest 重複填一次。

## 效能

Jetson Nano CPU（onnxruntime，`CPUExecutionProvider`）：

| 輸入尺寸 | 單次推理 | 說明 |
|---|---|---|
| 640（預設） | **約 12 秒** | 與 PyTorch 參考輸出逐項一致 |
| 448 | 約 6.3 秒 | 已一併匯出為 `models/demo/best_448.onnx` |

640 是訓練尺寸，準確度最可信。若要改用 448，把 manifest 的
`model_file` 改成 `models/demo/best_448.onnx`、`inference.image_size` 改成 448，
**並先用真實器械確認準確度可接受**——兩種尺寸的偵測結果並不相同。

## 器械名稱

`class_weight.json` 與 `surgery_instruments.json` 的器械名稱必須與模型輸出的
class name 逐字元相同（目前完全吻合，並有測試鎖住）。
不吻合時，選擇該手術類型會被明確拒絕，而不是默默顯示成「缺少」。
