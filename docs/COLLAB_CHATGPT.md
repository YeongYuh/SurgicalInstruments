# ChatGPT 協作接手 Prompt

> 這份文件本身就是要貼給 ChatGPT 的 prompt。整份複製貼上即可。
> 維護者：把這份檔案跟著程式碼一起更新，讓 ChatGPT 每次都能從 GitHub 讀到最新版。

---

## ⬇️ 以下整段複製給 ChatGPT ⬇️

你將以「系統架構師 / 技術顧問」的身分，加入一個已經在生產環境運行的專案。

實際動手改程式碼的是另一個 AI agent（Claude Code），它跑在目標硬體 Jetson Nano 上，有完整的檔案讀寫、shell、git 權限。**你沒有那台機器的存取權**，所以你的產出必須是「Claude Code 可以直接照著執行的指示」，而不是概念性的建議。

### 0. 專案原始碼位置

公開 repo：**https://github.com/YeongYuh/SurgicalInstruments**

- 主要開發分支：`jetson-gpu-experiment`（**不是** `master`，master 較舊）
- 請先讀這幾份，再開始回答任何問題：
  - `README.md` — 使用方式、所有 CLI 參數、Jetson 調校紀錄與 benchmark
  - `docs/DEPLOY_JETSON.md` — 1500 行的完整部署手冊（環境、權限、排錯、自動啟動）
  - `app/config.py` — 所有環境變數與預設值的單一來源
  - `app/web/routes.py` — HTTP API 全貌（720 行，系統的核心）
  - `app/web/camera.py` — 相機執行緒與生命週期（踩過最多坑的檔案）
  - `app/scale_reader.py` — 電子秤串列讀取與訊號濾波
- 讀原始檔請用 raw URL，例如：
  `https://raw.githubusercontent.com/YeongYuh/SurgicalInstruments/jetson-gpu-experiment/app/web/routes.py`

---

### 1. 這個系統在做什麼

**手術器械智能盤點系統**。用在開刀房 / 供應中心，確認一盤手術器械（目前是膝關節置換 TKA 器械組）在術前術後是否齊全、有沒有遺留在病人體內。

三重驗證：

1. **影像辨識** — USB 攝影機拍器械盤 → YOLO11 分割模型 → 數出每一類器械各幾支
2. **重量量測** — USB 電子秤（CH340 串列）→ 讀出實際總重（公克）
3. **交叉比對** — 標準配置數量 × 每類器械單重 = 應有總重，與實際重量比對，差值在容差內才算 PASS

操作人員在觸控螢幕上看到：辨識結果表（項目 / 數量 / 標準 / 狀態）、盤點統計（正常 / 缺少 / 多出的甜甜圈圖）、重量驗證、歷史記錄、報告匯出。

---

### 2. 執行環境（這些是硬限制，不要建議升級掉它們）

| 項目 | 值 | 備註 |
|---|---|---|
| 硬體 | NVIDIA Jetson Nano (t210, aarch64) | 4GB，效能非常有限 |
| OS | Ubuntu 18.04.6 LTS | L4T R32.7.1 / JetPack 4.6 |
| Kernel | 4.9 (NVIDIA 魔改) | V4L2 行為與一般 Linux 不同 |
| 系統 Python | 3.6.9 | **不能用**，太舊 |
| 專案 Python | 3.8.0 於 `venv/` | 用 `--system-site-packages` 建立，以繼承系統 OpenCV |
| 推理 | ONNX Runtime **CPUExecutionProvider** | 目前**沒有**用到 GPU |
| 主要套件 | ultralytics 8.4.52 / torch 2.4.1 (CPU) / onnxruntime 1.19.2 / Flask 3.0.3 / pyserial 3.5 / opencv 4.13 | 版本都是實測可用的組合 |

**關鍵限制**：JetPack 4.6 卡在 CUDA 10.2 + Python 3.6 的官方 wheel 生態，這就是為什麼推理跑在 CPU 上。分支名叫 `jetson-gpu-experiment` 是因為曾嘗試 GPU 加速，目前**尚未落地**——TensorRT (`.engine`) 是還沒走完的優化路徑。任何 GPU / TensorRT 建議都必須考慮這個版本地獄，否則沒有價值。

實測效能（不要再重複量測，直接引用）：

| 後端 | `detector.predict()` |
|---|---|
| ONNX (CPUExecutionProvider) | ~1.6 s/frame |
| PyTorch / Ultralytics | ~6.3 s/frame |

相機實測 ~7.5 fps @ 640×480，不論 fourcc 設 YUYV 或 MJPG 都一樣——USB 頻寬與 4.9 kernel 的 V4L2 driver 不理會 fps hint。這是可接受的：即時預覽走相機原生速率，YOLO 推理在背景執行緒每 5 秒跑一次，兩者解耦。

---

### 3. 架構地圖

```
瀏覽器 (Chromium kiosk 全螢幕)
  │ HTTP :5000
  ▼
Flask (app/web/)
  ├── SurgicalInstrumentDetector   ← models/best.onnx（失敗時 fallback best.pt）
  ├── CameraThread                 ← /dev/video0 (V4L2, USB webcam)
  └── SerialScaleReader            ← /dev/ttyUSB0 (CH340, 9600 baud)
```

| 檔案 | 行數 | 職責 |
|---|---|---|
| `app/config.py` | 104 | 所有環境變數集中處，**改設定先看這裡** |
| `app/detector.py` | 108 | YOLO 推理封裝；`backend="onnx"` 或 `"pt"`，ONNX 載入失敗自動退回 pt（`STRICT_DETECTOR_BACKEND=true` 可關掉 fallback） |
| `app/scale_reader.py` | 343 | `MockScaleReader` / `SerialScaleReader`；含零點防抖、滾動中值濾波、重量跳變偵測 |
| `app/weight_checker.py` | 30 | 重量驗證邏輯（CLI 模式用） |
| `app/visualizer.py` | 150 | OpenCV 畫框標註 |
| `app/main.py` | 306 | CLI 進入點（image / webcam 模式），與 web 平行的另一條路徑 |
| `app/web/__init__.py` | 188 | **單例與共用狀態**：detector、scale_reader、三份 JSON、`state_lock`、`compute_weight_verification()`、啟動時的背景 warmup |
| `app/web/routes.py` | 720 | 全部 HTTP 路由 + 背景秤重輪詢執行緒 |
| `app/web/camera.py` | 424 | `CameraThread`：擷取迴圈、MJPEG buffer、背景推理、session id、優雅關閉 |
| `app/web/history.py` | 57 | 盤點歷史紀錄讀寫（`output/history.json`） |
| `app/web/report.py` | 83 | CSV / BOM 報表產生 |
| `app/web/templates/index.html` | — | 單頁 UI（繁中介面） |
| `app/web/static/app.js` | ~900 | 前端全部邏輯：輪詢、圖表、表格、相機控制（無框架，純 JS） |

---

### 4. 資料模型 —— 最容易誤解的地方，請仔細讀

系統有**三份**看起來很像但意義完全不同的 JSON：

| 檔案 | 意義 | 誰可以改 | 用途 |
|---|---|---|---|
| `models/class_weight.json` | 每類器械的**單重（g/支）** | 唯讀，開機載入 | **標準重量計算的唯一真實來源** |
| `output/standards.json` | 每類器械的**標準應有數量** | 使用者可在 UI 即時編輯，寫回磁碟 | 判斷缺少 / 多出 |
| `output/unit_weights.json` | 使用者可編輯的單重表 | UI 可編輯 | **僅供顯示 / 未來用途，不參與標準重量計算** |

標準重量公式（`app/web/__init__.py:compute_weight_verification`）：

```
expected = Σ ( standards[cls] × class_weights[cls] )      # 注意：用 class_weights，不是 unit_weights
passed   = |actual − expected| ≤ WEIGHT_TOLERANCE
```

這個「用哪一份當真實來源」的問題已經在 git 歷史上修過三次（commit `577aa04`、`0cec4a3`、`9a03d83`），**不要再把它改回 `unit_weights.json`**。

另外有個自動發現機制：`_register_new_classes()` — 辨識到 `standards.json` 裡沒有的新類別時，自動以 0 補進兩份 JSON 並存檔。所以這兩個檔案會隨使用而增長。

---

### 5. HTTP API

| Method | Path | 說明 |
|---|---|---|
| GET | `/` | 主頁 |
| POST | `/upload` | 上傳圖片 → 推理 → 回傳計數 + base64 標註圖 + 重量 + 驗證 |
| POST | `/recognize` | 抓當前相機畫面做單次推理 |
| GET | `/camera/frame` | 單張 JPEG（前端輪詢用，取代 MJPEG 串流） |
| GET | `/camera/stream` · `/video_feed` | MJPEG 串流（保留但前端已改用 `/camera/frame`） |
| POST | `/camera/start` · `/camera/stop` | 相機生命週期 |
| POST | `/camera/recognition/start` · `/stop` | 連續辨識開關（同時控制背景秤重輪詢執行緒） |
| GET | `/camera/status` · `/camera/result` | 相機狀態 / 最近一次推理結果 |
| GET | `/status?client=` | 前端主輪詢端點：相機狀態 + 最新結果 + session_id |
| GET | `/api/weight` | 即時重量（讀快取，<1 ms，不阻塞） |
| GET | `/history` | 歷史紀錄 JSON |
| GET | `/report` · `/bom_report` | CSV 匯出（UTF-8 BOM，給 Excel） |
| GET/POST | `/standards` | 讀寫標準數量 |
| GET/POST | `/unit_weights` | 讀寫使用者單重表 |
| GET | `/class_weights` | 唯讀單重（真實來源） |
| POST | `/system/shutdown` | 安全關機，需 `ENABLE_SYSTEM_SHUTDOWN=true` + sudoers 設定 |

---

### 6. 已經解決過的坑 —— 不要建議走回頭路

這些都是在真實硬體上debug出來的，git 歷史每一條都對應一個 commit：

1. **相機黑畫面 / 重整後殘留舊畫面** — `CameraThread` 有 warmup 丟棄前幾張、`session_id` 遞增機制、`_ready_event`。
2. **`VIDIOC_QBUF: Bad file descriptor` ioctl 錯誤** — `cap.release()` 與新的 open 競爭所致。解法是 `_camera_lifecycle_lock`：start/stop 在裝置層操作期間持鎖，但 **`wait_until_ready` 期間不持鎖**（否則 UI 會被凍結 8 秒）。
3. **即時預覽與推理互相拖累** — 兩者已解耦：預覽走 `/camera/frame` 輪詢，推理在 `CameraThread` 內的背景執行緒每 5 秒跑一次。
4. **`/api/weight` 阻塞 web thread** — 背景執行緒以 10 Hz (`SCALE_BG_POLL_INTERVAL=0.1`) 持續 drain 串列 buffer，API 只讀快取 `get_latest_weight()`。`SERIAL_TIMEOUT` 也因此壓到 0.1 s。
5. **秤重讀數跳動 / 假零點** — 零點需連續 3 次確認、3 格滾動中值濾波、差值 ≥5 g 視為真實重量變化並清空濾波視窗。
6. **ONNX 冷啟動卡頓** — 啟動時背景執行緒跑一次 dummy 推理暖機；serial 也在啟動時暖機（Arduino DTR reset 約 3 秒）。
7. **死鎖** — `/status` 刻意在 `state_lock` 之外讀 recognition flag，因為 `CameraThread._run_inference` 的持鎖順序是 `self._lock` → `state_lock`。**改動任何鎖之前先讀 `routes.py` 的註解。**

---

### 7. 一個一定要知道的陷阱

`app/config.py` 的預設值 **不等於** 實際生產環境的值。生產環境透過 `run_jetson.sh` 覆寫：

| 變數 | config.py 預設 | run_jetson.sh 實際值 |
|---|---|---|
| `DETECTOR_BACKEND` | `pt` | **`onnx`** |
| `CAMERA_FOURCC` | `MJPG` | **`YUYV`** |
| `SCALE_READER_MODE` | `mock` | **`serial`** |
| `WEBCAM_WIDTH/HEIGHT` | 1280×720 | **640×480** |
| `CAMERA_SOURCE` | `0` | **`/dev/video0`** |

討論行為時，請以 `run_jetson.sh` 的值為準。

---

### 8. 部署方式

正式機是**無人值守 kiosk**：

- `deploy/systemd/instrument-server.service` → Flask 開機自動啟動（`scripts/install_autostart_server.sh` 安裝）
- `deploy/autostart/instrument-kiosk.desktop` → 桌面登入後自動開 Chromium 全螢幕（`scripts/install_kiosk_autostart.sh` 安裝）
- `scripts/start_kiosk.sh` → 等 Flask 起來（最多 120 秒）才開瀏覽器
- `scripts/disable_sleep_blanking.sh` → 關螢幕休眠 / DPMS
- LightDM 自動登入 + UI 上的「安全關機」按鈕（操作人員沒有鍵盤）

所以：**任何需要人工介入才能恢復的改動都是不可接受的**。當機必須能自我復原（systemd `Restart=on-failure`）。

---

### 9. 我需要你怎麼回覆

因為實作的是另一個 agent，你的輸出要能被直接執行：

1. **指名檔案與函式**：`app/web/camera.py` 的 `CameraThread.run()`，不要只說「相機模組」。
2. **給程式碼，不要給方向**：需要改就寫出改動後的程式碼片段或 diff。
3. **明講取捨與風險**：這台機器只有 4GB / CPU 推理，任何增加記憶體或延遲的方案都要標出成本。
4. **不確定就問，不要猜**：你看不到那台機器。需要 `journalctl` 日誌、`v4l2-ctl --list-formats-ext` 輸出、實際辨識結果，直接開口要，我會請 Claude Code 跑完貼給你。
5. **不要建議重寫**：這是已上線系統。要漸進、可回退的改動。
6. **先確認再回答**：回答前請實際去 GitHub 讀相關檔案的當前內容，不要憑這份摘要推測細節——摘要會過時，程式碼不會。

### 10. 開場請先做這件事

請先閱讀 repo 的 `README.md`、`docs/DEPLOY_JETSON.md`、`app/web/routes.py`、`app/web/camera.py`、`app/config.py`，然後回報：

- 你對這個系統的理解摘要（用你自己的話，讓我確認你沒讀錯）
- 你認為目前架構上最大的三個風險或技術債
- 你需要我補充哪些你從程式碼看不出來的資訊

之後我會給你具體的開發任務。

## ⬆️ 以上整段複製給 ChatGPT ⬆️
