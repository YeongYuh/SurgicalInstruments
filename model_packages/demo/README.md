# Demo 器械套件（範本）

`"template": true` 表示這是尚未填寫的範本：系統會列出它，但拒絕啟用，
也不會驗證下列檔案是否存在。填完後請把 `"template"` 改成 `false`。

實際需要放入的東西：

| 項目 | 放置路徑 | 說明 |
|---|---|---|
| 模型檔 | `models/demo/model.onnx` | 或改 manifest 的 `model_file` 指到你的實際路徑 |
| 備援模型檔 | `models/demo/model.pt` | 選用；主檔不存在時才使用 |
| 每類單重 | `model_packages/demo/class_weight.json` | `{"類別名稱": 公克數, ...}`，類別名稱必須與模型輸出一致 |
| 標準數量 | `model_packages/demo/standards.json` | `{"類別名稱": 應有數量, ...}`，出廠預設值 |

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
