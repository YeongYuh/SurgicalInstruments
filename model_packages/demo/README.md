# Demo 器械套件（範本）

`"template": true` 表示這是尚未填寫的範本：系統會列出它，但拒絕啟用，
也不會驗證下列檔案是否存在。填完後請把 `"template"` 改成 `false`。

實際需要放入的東西：

| 項目 | 放置路徑 | 說明 |
|---|---|---|
| 模型檔 | `models/demo/model.onnx` | 目錄已建好，且該目錄下的模型檔已被 `.gitignore` 忽略，不會誤 commit |
| 備援模型檔 | `models/demo/model.pt` | 選用；主檔不存在時才使用 |
| 每類單重 | `model_packages/demo/class_weight.json` | `{"類別名稱": 公克數, ...}`，類別名稱必須與模型輸出一致 |
| 標準數量 | `model_packages/demo/standards.json` | `{"類別名稱": 應有數量, ...}`，出廠預設值 |

已隨套件附上（不需你再建立）：

| 檔案 | 內容 |
|---|---|
| `class_weight.json` | 17 種器械的實際單重（公克） |
| `surgery_instruments.json` | SurgeryA–D 四種手術的標準器械盤 |

> **手術類型的器械名稱必須與模型輸出的 class name 逐字元相同。**
> 不相同時，選擇該手術類型會被拒絕（因為那盤器械永遠不可能點齊），
> 而不是默默顯示成「缺少」。拿到模型後第一件事是比對 class names。

> 只有模型、還沒有重量與標準數量時：可先讓 `standards.json` 全部填 0，
> `class_weight.json` 只填真正有資料的類別。辨識與盤點可以先跑，重量驗證會顯示
> 「尚未設定標準重量」而不是假的 PASS/FAIL。**不要為了讓它變綠色而編數字。**

manifest 需要確認的欄位：

- `adapter` — 目前只有 `ultralytics`。若模型不是 Ultralytics 匯出的，
  必須先新增對應 adapter 並用 `register_adapter()` 註冊，再把名稱填在這裡。
  **不要**因為副檔名是 `.onnx` 就沿用 `ultralytics`：ONNX 只是序列化格式，
  不同模型的輸出張量與後處理完全不同。
- `adapter_options.task` — `detect` / `segment` / `classify`，ONNX 必填。
- `inference.confidence` / `inference.image_size`
- `template` → 改為 `false`

完成後：

```bash
curl -s localhost:5000/api/model-packages          # 確認 valid=true
curl -s -X POST localhost:5000/api/model-package \
     -H 'Content-Type: application/json' -d '{"id":"demo"}'
```

首次啟用時，`model_packages/demo/standards.json` 會被複製到
`output/profiles/demo/standards.json`，之後在畫面上調整的標準數量只會寫入
profile，不會動到套件內的出廠預設值。
