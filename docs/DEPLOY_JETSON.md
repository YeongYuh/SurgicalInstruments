# Jetson Nano 部署指南

手術器械智能盤點系統 — 全新機器從零開始設定

> **適用環境**：Jetson Nano · Ubuntu 18.04.6 LTS · L4T R32.x / JetPack 4.x · ONNX CPU 推理

---

## 目錄

1. [系統概覽](#1-系統概覽)
2. [硬體需求](#2-硬體需求)
3. [OS 環境假設](#3-os-環境假設)
4. [取得專案（從 GitHub 下載並安裝）](#4-取得專案)
5. [安裝系統套件](#5-安裝系統套件)
6. [建立或修復 Python venv](#6-建立或修復-python-venv)
7. [Python 依賴說明](#7-python-依賴說明)
8. [模型檔案](#8-模型檔案)
9. [攝影機設定](#9-攝影機設定)
10. [電子秤串列埠設定](#10-電子秤串列埠設定)
11. [環境變數](#11-環境變數)
12. [啟動應用程式](#12-啟動應用程式)
13. [安全關機按鈕設定](#13-安全關機按鈕設定)
14. [瀏覽器操作說明](#14-瀏覽器操作說明)
15. [Kiosk / 觸控螢幕模式](#15-kiosk--觸控螢幕模式)
16. [開機自動啟動與 Kiosk 全螢幕](#16-開機自動啟動與-kiosk-全螢幕)
17. [驗證清單](#17-驗證清單)
18. [常見問題排查](#18-常見問題排查)
19. [Git 注意事項](#19-git-注意事項)
20. [最終快速驗證指令](#20-最終快速驗證指令)

---

## 1. 系統概覽

本系統為手術器械智能盤點系統，在 Jetson Nano 上執行 Flask Web 應用程式，提供以下功能：

| 功能 | 說明 |
|------|------|
| 器械辨識 | 以 YOLO 模型透過 ONNX Runtime（CPU）推理，辨識手術器械種類與數量 |
| 即時預覽 | USB 網路攝影機 MJPEG 串流，低延遲即時顯示 |
| 重量量測 | 透過 Arduino/CH340 USB 串列接口讀取電子秤數值 |
| 標準重量 | 依 `models/class_weight.json` 中每類器械的單位公克數計算應有總重 |
| Web UI | Flask 提供瀏覽器介面，支援圖片上傳辨識、攝影機辨識、盤點統計、歷史記錄、報告匯出 |

系統架構：

```
瀏覽器 (Firefox / Chromium)
  │  HTTP (Flask :5000)
  ▼
Flask 後端
  ├── ONNX Runtime (CPUExecutionProvider) ← models/best.onnx
  ├── CameraThread  ← /dev/video0 (USB 攝影機)
  └── SerialScaleReader ← /dev/ttyUSB0 (CH340 電子秤)
```

---

## 2. 硬體需求

| 硬體 | 規格 / 說明 |
|------|-------------|
| Jetson Nano | Developer Kit（4 GB 或 2 GB，aarch64） |
| 儲存媒體 | microSD ≥ 32 GB（建議 Class 10 / U3）或 NVMe（附轉接座） |
| USB 網路攝影機 | 預設裝置路徑 `/dev/video0`（已測試：Logitech C310） |
| 電子秤 | 透過 Arduino + 稱重模組，USB CH340 轉接器連接（`lsusb` ID `1a86:7523`），預設路徑 `/dev/ttyUSB0` |
| HDMI 顯示器 | 可選，7 吋 1024×600 顯示器（UI 已針對此尺寸調整），也可透過 LAN 從其他電腦操作 |
| 網路連線 | LAN 有線或 Wi-Fi，建議有線以降低延遲 |
| 電源供應 | 5V 4A 變壓器（DC 桶形插頭），電流不足會導致攝影機與串列裝置不穩定 |

---

## 3. OS 環境假設

- **作業系統**：Ubuntu 18.04.6 LTS（Bionic Beaver）
- **核心**：L4T R32.x（JetPack 4.x，例如 R32.7.1 JetPack 4.6.1）
- **架構**：aarch64
- **系統 Python**：3.6.9（Ubuntu 18.04 預設；本專案**不使用**此版本）
- **本專案 Python**：**3.8**（需手動安裝，見第 5 節）
- **CUDA**：本專案 ONNX CPU 路徑**不需要 CUDA**；PyTorch CUDA 功能不啟用
- **GPU 推理**（TensorRT）：本指南不涵蓋，請勿隨意嘗試

> **重要**：JetPack 4.x 搭配 PyTorch 2.x 的 CUDA 10.2 相容性有問題，因此本系統固定使用 ONNX CPU 推理。

---

## 4. 取得專案

本節說明如何在全新 Jetson Nano 上從 GitHub 下載並安裝本專案，以及如何使用已複製的目錄。

---

### 從 GitHub 下載並安裝

#### 前提條件

- Jetson 必須能連上網路（有線或 Wi-Fi）。
- Git 必須已安裝：

  ```bash
  sudo apt update
  sudo apt install -y git
  ```

- 確認 Git 版本：

  ```bash
  git --version
  # 預期：git version 2.x.x
  ```

#### 選擇安裝目錄

建議使用以下路徑：

```bash
mkdir -p /home/camlion/projects
cd /home/camlion/projects
```

> 若使用不同使用者名稱，請將 `camlion` 替換為實際使用者名稱。

---

#### 方法 A：HTTPS 克隆（推薦用於全新機器）

HTTPS 克隆不需要設定 SSH 金鑰，適合快速取得專案。

```bash
git clone https://github.com/YeongYuh/SurgicalInstruments.git instrument
cd instrument
git checkout jetson-gpu-experiment
```

**關於認證**：

- 若儲存庫為**公開（public）**，`git clone` 可直接執行，不需登入。
- 若儲存庫為**私有（private）**，Git 會要求輸入使用者名稱與密碼。
  > ⚠️ **注意**：GitHub 已停止接受帳號密碼進行 Git 操作。請改用 **Personal Access Token（個人存取令牌）** 作為密碼欄位的輸入。

**取得 Personal Access Token**：

1. 前往 GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 點擊 **Generate new token**
3. 設定 Scope：至少勾選 `repo`
4. 複製產生的 token（只顯示一次）
5. 在 `git clone` 提示輸入密碼時，貼上此 token

---

#### 方法 B：SSH 克隆

SSH 克隆不需要每次輸入認證資訊，適合長期使用的機器。

**步驟 1：確認 SSH 金鑰是否已設定**

```bash
ssh -T git@github.com
```

- 若顯示 `Hi <username>! You've successfully authenticated...`，代表已設定完成，跳至步驟 4。
- 若顯示 `Permission denied (publickey)`，需先產生金鑰（步驟 2）。

> 若首次連線 GitHub，會出現以下提示：
> ```
> The authenticity of host 'github.com' can't be established.
> Are you sure you want to continue connecting (yes/no)?
> ```
> 輸入 `yes` 即可，GitHub 的主機金鑰會被加入 `~/.ssh/known_hosts`。

**步驟 2：產生 SSH 金鑰**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- 按 Enter 接受預設路徑（`~/.ssh/id_ed25519`）
- 可設定 passphrase 或直接 Enter 略過

**步驟 3：將公開金鑰加入 GitHub**

```bash
cat ~/.ssh/id_ed25519.pub
```

複製輸出的完整內容，然後：

1. 開啟瀏覽器，前往 `https://github.com/settings/keys`
2. 點擊 **New SSH key**
3. Title：填入識別名稱（例如 `Jetson Nano`）
4. Key type：選擇 **Authentication Key**
5. Key：貼上剛才複製的公開金鑰
6. 點擊 **Add SSH key**

**步驟 4：SSH 克隆**

```bash
git clone git@github.com:YeongYuh/SurgicalInstruments.git instrument
cd instrument
git checkout jetson-gpu-experiment
```

---

#### 快速指令整合（HTTPS 路徑）

以下為從零開始的完整命令序列：

```bash
sudo apt update
sudo apt install -y git
mkdir -p /home/camlion/projects
cd /home/camlion/projects
git clone https://github.com/YeongYuh/SurgicalInstruments.git instrument
cd instrument
git checkout jetson-gpu-experiment
git status
```

---

#### 確認儲存庫狀態

克隆完成後，確認以下項目：

```bash
git status
git branch
git log --oneline -5
git remote -v
```

**預期輸出**：

```
On branch jetson-gpu-experiment
Your branch is up to date with 'origin/jetson-gpu-experiment'.
nothing to commit, working tree clean

* jetson-gpu-experiment
  master

origin  https://github.com/YeongYuh/SurgicalInstruments.git (fetch)
origin  https://github.com/YeongYuh/SurgicalInstruments.git (push)
```

- Branch 應為 `jetson-gpu-experiment`
- Remote 應指向 `YeongYuh/SurgicalInstruments`
- 工作目錄應為乾淨（除非 `output/` 目錄的執行時期設定已被修改）

---

#### 克隆後確認模型檔案

> ⚠️ **重要**：大型模型檔（`best.pt`、`best.onnx`）可能被 `.gitignore` 排除，不包含在 Git 儲存庫中。克隆後請立即確認：

```bash
ls -lh models/
test -f models/best.pt          && echo "best.pt OK"          || echo "best.pt 缺失"
test -f models/best.onnx        && echo "best.onnx OK"        || echo "best.onnx 缺失"
test -f models/class_weight.json && echo "class_weight.json OK" || echo "class_weight.json 缺失"
```

| 檔案 | 若缺失的處理方式 |
|------|-----------------|
| `models/best.pt` | 從備份機器手動複製，或洽詢專案維護者 |
| `models/best.onnx` | 若 `best.pt` 存在，可重新匯出（見第 8 節） |
| `models/class_weight.json` | 應包含於 Git，若缺失請確認 branch 正確 |

**請勿期望 GitHub 儲存庫一定包含大型二進位模型檔。**

---

#### 克隆後繼續設定

取得專案後，繼續依序完成：

1. [第 5 節](#5-安裝系統套件)：安裝系統套件（Python 3.8、libgl1 等）
2. [第 6 節](#6-建立或修復-python-venv)：建立 Python 3.8 venv
3. 安裝 Python 依賴：`pip install -r requirements.txt`
4. [第 9 節](#9-攝影機設定)：設定攝影機權限
5. [第 10 節](#10-電子秤串列埠設定)：設定電子秤串列埠權限
6. [第 12 節](#12-啟動應用程式)：執行 `./run_jetson.sh`

---

#### GitHub 克隆常見問題

**A. `Permission denied (publickey)`**

- **原因**：Jetson 的 SSH 金鑰未加入 GitHub，或使用了錯誤的 clone URL。
- **修復**：改用 HTTPS 克隆，或依照方法 B 設定 SSH 金鑰。

**B. 輸入使用者名稱/密碼後被拒（`Authentication failed`）**

- **原因**：GitHub 已停止接受帳號密碼進行 Git 操作（2021 年起）。
- **修復**：在密碼欄位輸入 Personal Access Token，而非帳號密碼。

**C. `repository not found`**

- **原因**：URL 有誤，或私有儲存庫沒有存取權限。
- **修復**：確認 URL 拼寫正確；若為私有儲存庫，確認帳號已被授予存取權，並使用正確的認證方式。

**D. 首次連線出現主機金鑰確認提示**

```
The authenticity of host 'github.com' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

- 輸入 `yes` 繼續，GitHub 的主機金鑰會被記錄至 `~/.ssh/known_hosts`，下次不再出現此提示。

---

### 方法 B：使用已複製的目錄

若專案目錄已透過 USB 隨身碟或 `rsync` 從其他機器複製，進入目錄後確認 git 狀態：

```bash
cd /home/camlion/projects/instrument
git status
git branch
git remote -v
```

若顯示的 branch 正確，即可繼續。若有未預期的修改，請視情況決定是否還原。

> 若 venv 也一併複製過來，請見第 6 節方法 B 的修復步驟（venv 路徑可能仍指向舊機器的使用者目錄）。

---

## 5. 安裝系統套件

```bash
sudo apt update
sudo apt install -y \
  python3.8 \
  python3.8-venv \
  python3.8-distutils \
  git \
  curl \
  libgl1 \
  libglib2.0-0
```

- `python3.8`：本專案 venv 所需，Ubuntu 18.04 預設只有 3.6，需手動安裝
- `python3.8-venv`：建立虛擬環境所需
- `python3.8-distutils`：pip 安裝部分套件時需要
- `git`：從 GitHub 下載專案所需
- `curl`：**`scripts/start_kiosk.sh` 的必要依賴**，用於等待 Flask 伺服器就緒再開啟 Chromium；若缺少 curl，Kiosk 腳本會立即失敗
- `libgl1` / `libglib2.0-0`：OpenCV 執行時期依賴（無 GUI 環境常缺少）

**選用的診斷工具與 Kiosk 工具**：

```bash
sudo apt install -y v4l-utils usbutils unclutter
```

- `v4l-utils`：提供 `v4l2-ctl` 指令，可查詢攝影機支援格式
- `usbutils`：提供 `lsusb` 指令，可確認 USB 裝置識別
- `unclutter`：Kiosk 模式下自動隱藏滑鼠游標（閒置 1 秒後消失）；`start_kiosk.sh` 若偵測到 `unclutter` 即自動啟用，未安裝時略過

**確認 Python 3.8 安裝成功**：

```bash
python3.8 --version
# 預期：Python 3.8.0
```

---

## 6. 建立或修復 Python venv

### 方法 A：全新建立 venv（推薦用於全新機器）

```bash
cd /home/camlion/projects/instrument

# 建立 venv（啟用 system-site-packages 以繼承 Jetson 系統 OpenCV）
python3.8 -m venv venv

source venv/bin/activate

# 確認使用的是 venv 內的 Python 3.8
which python3
# 預期：/home/camlion/projects/instrument/venv/bin/python3
python3 --version
# 預期：Python 3.8.x

# 升級 pip / setuptools / wheel
pip install --upgrade pip setuptools wheel

# 安裝所有依賴
pip install -r requirements.txt
```

> **注意**：`requirements.txt` 不包含 `torch`、`onnxruntime`、`opencv-python`，  
> 因為這些在 Jetson Nano 上有特定的 aarch64 wheel，需要另行安裝（見下方）。

若為全新機器且 venv 中還沒有 torch / onnxruntime / opencv-python，需補充安裝：

```bash
# opencv-python（aarch64 wheel）
pip install opencv-python

# onnxruntime（1.16–1.19 為 Jetson aarch64 可用版本）
pip install "onnxruntime>=1.16,<1.20"

# torch 2.4.1 CPU wheel（aarch64，由專案驗證可用）
# 請使用專案原始 venv 中的版本，或洽詢專案維護者取得 wheel 檔
# torch 版本務必與 ultralytics==8.4.52 相容
pip install torch==2.4.1
```

### 方法 B：修復從其他機器複製來的 venv

若 venv 是從另一台機器（使用者 `ncut`）整個複製過來，`activate` 腳本和所有執行檔的 shebang 仍指向舊路徑，需要修復。

**步驟 1：確認是否有舊路徑殘留**

```bash
grep -R "/home/ncut/projects/instrument" -n venv/bin 2>/dev/null | head -20
```

若有輸出，代表需要修復。

**步驟 2：批次替換路徑**

```bash
grep -Rl "/home/ncut/projects/instrument" venv/bin | \
  xargs sed -i 's#/home/ncut/projects/instrument#/home/camlion/projects/instrument#g'
```

> 若使用不同的使用者名稱，請將 `camlion` 換成實際使用者名稱。

**步驟 3：修復 Python 3.8 符號連結**

```bash
ln -sf /usr/bin/python3.8 venv/bin/python3.8
ln -sf python3.8 venv/bin/python3
ln -sf python3.8 venv/bin/python
```

**步驟 4：驗證修復結果**

```bash
source venv/bin/activate

which python3
# 預期：/home/camlion/projects/instrument/venv/bin/python3

python3 --version
# 預期：Python 3.8.x

pip --version
# 預期：pip x.x from /home/camlion/.../venv/lib/python3.8/...

# 快速 import 驗證
python3 -c "import flask, cv2, numpy, ultralytics, onnxruntime, serial, torch; print('all OK')"
```

### 方法 C：以 --upgrade 原地重建 venv 基礎設施

若修復步驟 B 後仍有問題，可用 `--upgrade` 重建 venv 的 bin 與 pyvenv.cfg，不刪除已安裝套件：

```bash
python3.8 -m venv --upgrade /home/camlion/projects/instrument/venv
```

---

## 7. Python 依賴說明

### 重要原則

> **請勿隨意升級以下套件**（版本對 Jetson aarch64 有特定相容性）：
> - `ultralytics`（固定 8.4.52）
> - `torch`（固定 2.4.1 CPU）
> - `opencv-python`（已安裝版本）
> - `numpy`（已安裝版本）
> - `onnxruntime`（已安裝版本）

### 已知需要手動補裝的依賴

若複製的 venv 缺少以下套件（這些套件在原始機器上是系統 Python 3.6 的套件，不在 Python 3.8 的路徑內），請逐一安裝：

```bash
# requests 的傳遞依賴（ultralytics 需要 requests）
pip install six urllib3 certifi idna
```

### matplotlib / ultralytics 圖表依賴

若出現 matplotlib 相關的 `ModuleNotFoundError`，請只安裝缺少的項目：

```bash
pip install cycler kiwisolver pyparsing packaging pillow contourpy fonttools python-dateutil
```

### 安裝後更新 requirements.txt

安裝任何新的缺少套件後，請將套件名稱加入 `requirements.txt`（使用未鎖定版本，與現有格式一致），並提交到 git。

### 完整 import 測試指令

```bash
python3 - <<'PY'
imports = [
    ("flask", "flask"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("ultralytics", "ultralytics"),
    ("onnxruntime", "onnxruntime"),
    ("serial", "pyserial"),
    ("torch", "torch"),
    ("matplotlib", "matplotlib"),
    ("cycler", "cycler"),
    ("kiwisolver", "kiwisolver"),
    ("pyparsing", "pyparsing"),
    ("packaging", "packaging"),
    ("PIL", "pillow"),
    ("contourpy", "contourpy"),
    ("fontTools", "fonttools"),
    ("dateutil", "python-dateutil"),
    ("six", "six"),
    ("urllib3", "urllib3"),
    ("certifi", "certifi"),
    ("idna", "idna"),
    ("requests", "requests"),
]
missing = []
for module, package in imports:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "unknown")
        print(f"[OK] {module} ({package}) {ver}")
    except Exception as e:
        print(f"[MISSING] {module} ({package}): {e}")
        missing.append(package)
print()
print("MISSING_PACKAGES=", " ".join(sorted(set(missing))) or "(none)")
PY
```

---

## 8. 模型檔案

### 必要檔案

| 檔案 | 說明 |
|------|------|
| `models/best.pt` | 原始 Ultralytics YOLO 模型（PyTorch 格式，約 20 MB） |
| `models/best.onnx` | ONNX 格式推理模型（預設執行時使用，約 39 MB） |
| `models/class_weight.json` | 每類器械的單位重量（公克），標準重量計算的唯一來源 |

> **注意**：`*.pt` 和 `*.onnx` 是大型二進位檔，通常被 `.gitignore` 排除，不包含在 git 倉庫中。部署時需手動複製或從其他來源取得。`class_weight.json` 應包含在 git 中。

### 確認檔案存在

```bash
ls -lh models/
test -f models/best.pt    && echo "[OK] best.pt"    || echo "[MISSING] best.pt"
test -f models/best.onnx  && echo "[OK] best.onnx"  || echo "[MISSING] best.onnx"
test -f models/class_weight.json && echo "[OK] class_weight.json" || echo "[MISSING] class_weight.json"
```

### 從 best.pt 重新匯出 best.onnx

若 `best.onnx` 缺失，可從 `best.pt` 重新匯出：

```bash
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("models/best.pt")
model.export(format="onnx", opset=12, imgsz=640)
print("匯出完成：models/best.onnx")
PY
```

### 確認模型類別

```bash
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("models/best.pt")
print("類別數量:", len(model.names))
for k, v in model.names.items():
    print(f"  {k}: {v}")
PY
```

### 器械模型套件

模型不再寫死在程式裡，而是由套件描述：

```
model_packages/ortho_tka/manifest.json   # adapter、模型檔、推理參數、資料來源
model_packages/ortho_tka/standards.json  # 出廠預設標準數量（唯讀）
output/profiles/ortho_tka/standards.json # 現場調整後的標準數量（可編輯）
```

首次啟動時，若 `output/profiles/<id>/` 不存在，系統會依序從
「舊版全域 `output/standards.json`」→「套件出廠預設」種入，
所以既有機器升級後現場設定會保留。

確認與切換：

```bash
curl -s localhost:5000/api/model-packages | python3 -m json.tool   # 列出所有套件
curl -s localhost:5000/api/model-package  | python3 -m json.tool   # 目前使用中

# 切換（同步完成才回應，失敗時原套件仍可用）
curl -s -X POST localhost:5000/api/model-package \
     -H 'Content-Type: application/json' -d '{"id":"ortho_tka"}'
```

也可以直接在畫面右上角的「器械套件」下拉選單切換。

新增一個器械套件（例如 Demo 模型）不需要改任何 Python：

1. 放入模型檔（例如 `models/demo/model.onnx`）
2. 建立 `model_packages/demo/class_weight.json`（每類公克數）
3. 建立 `model_packages/demo/standards.json`（標準數量）
4. 編輯 `model_packages/demo/manifest.json`，把 `"template"` 改成 `false`
5. 用上面的 API 或下拉選單切換過去

詳細欄位說明見 `model_packages/demo/README.md`。

> `adapter` 必須明確指定。副檔名 `.onnx` **不代表**推理方式——
> 不同 ONNX 模型的輸出張量與後處理完全不同，所以系統絕不從副檔名猜測。
> 若模型不是 Ultralytics 匯出的，需新增一個 `ModelAdapter` 子類別並用
> `register_adapter()` 註冊，平台其他程式碼不用動。

---

## 9. 攝影機設定

### 預設值

| 參數 | 預設值 |
|------|--------|
| `CAMERA_SOURCE` | `/dev/video0` |
| `CAMERA_FOURCC` | `YUYV` |
| `WEBCAM_WIDTH` | `640` |
| `WEBCAM_HEIGHT` | `480` |
| `WEBCAM_FPS` | `15` |

### 確認裝置存在

```bash
ls -l /dev/video*
v4l2-ctl --list-devices 2>/dev/null || echo "(v4l-utils 未安裝)"
```

### 確認使用者有讀寫權限

```bash
# 使用者必須在 video 群組中
groups
# 若缺少 video 群組：
sudo usermod -aG video $USER
# 然後登出再登入
```

### 攝影機功能測試

```bash
source venv/bin/activate
python3 tools/test_camera_preview.py
```

### Jetson 特有說明

- Jetson 4.9 核心的 V4L2 驅動程式在設定 FOURCC 後回傳 `fourcc=0`，這是驅動程式特性，YUYV 設定仍有效，**不是錯誤**。
- 攝影機停止時，OpenCV 可能印出 `ioctl(VIDIOC_QBUF): Bad file descriptor`，這是 V4L2 緩衝區釋放的正常警告，只要攝影機能重新開啟即可忽略。
- **請勿**設定 `CAP_PROP_BUFFERSIZE=1`，會在 Jetson 核心上引發 QBUF 錯誤。

### 切換攝影機裝置

```bash
# 若攝影機在 /dev/video1
CAMERA_SOURCE=/dev/video1 ./run_jetson.sh
```

---

## 10. 電子秤串列埠設定

### 預設值

| 參數 | 預設值 |
|------|--------|
| `SCALE_READER_MODE` | `serial` |
| `SERIAL_PORT` | `/dev/ttyUSB0` |
| `SERIAL_BAUDRATE` | `9600` |

### 確認 CH340 裝置

```bash
lsusb | grep 1a86
# 預期看到：ID 1a86:7523 QinHeng Electronics HL-340 USB-Serial adapter

ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || echo "(無串列裝置)"
```

### 設定串列埠權限

```bash
sudo usermod -aG dialout $USER
```

> **重要**：執行後必須**登出再登入**（或重新開機）才會生效。

### 驗證權限

```bash
# 登出後重新登入，再執行：
groups
# 輸出中應包含 dialout

python3 - <<'PY'
import os
path = "/dev/ttyUSB0"
exists = os.path.exists(path)
rw = os.access(path, os.R_OK | os.W_OK)
print(f"{path}  exists={exists}  rw={rw}")
# 預期：exists=True  rw=True
PY
```

### 無電子秤的測試模式

```bash
SCALE_READER_MODE=mock ./run_jetson.sh
```

### 切換串列裝置

```bash
# 若電子秤在 /dev/ttyACM0（部分 Arduino 板）
SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh
```

---

## 11. 環境變數

`./run_jetson.sh` 已設定安全的預設值，通常不需要修改。若需要覆蓋，在指令前設定環境變數即可。

### 器械模型套件

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ACTIVE_MODEL_PACKAGE` | `ortho_tka` | 啟動時載入的器械套件 |
| `MODEL_PACKAGES_DIR` | `model_packages` | 套件目錄 |
| `PROFILES_DIR` | `output/profiles` | 各套件的現場設定（標準數量／單重） |
| `CONF_THRESHOLD` | 由 manifest 決定 | 設定時才覆蓋 manifest 的 `inference.confidence` |

> 模型檔、adapter、task、每類單重、預設標準數量一律由
> `model_packages/<id>/manifest.json` 決定。
> `DETECTOR_BACKEND` / `ONNX_MODEL_PATH` / `ONNX_TASK` 僅剩 `app/main.py`
> 這支獨立 CLI 使用，Web 平台已不再讀取。

### 重量穩定度

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SCALE_STABLE_WINDOW_SEC` | `1.5` | 判定穩定的時間視窗 |
| `SCALE_STABLE_RANGE_GRAMS` | `1.0` | 視窗內允許的最大波動（公克） |
| `SCALE_STABLE_MIN_SAMPLES` | `3` | 視窗內最少樣本數 |
| `SCALE_MAX_SAMPLE_AGE_SEC` | `2.0` | 超過此秒數的快取讀值視為 `fresh=false` |
| `SCALE_STABLE_MIN_COVERAGE_RATIO` | `0.5` | 樣本需覆蓋視窗的比例 |

> 重量未穩定前不會給出 PASS/FAIL，畫面顯示「量測中」而非紅色「不符合」。

### 攝影機自動復原

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `CAMERA_READ_FAIL_THRESHOLD` | `60` | 連續讀取失敗幾次後重開攝影機 |
| `CAMERA_REOPEN_BACKOFF_SEC` | `1.0` | 重開的起始退避秒數 |
| `CAMERA_REOPEN_MAX_BACKOFF_SEC` | `15.0` | 退避上限（不會忙碌迴圈） |

### 攝影機

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `CAMERA_SOURCE` | `/dev/video0` | 攝影機路徑或索引 |
| `CAMERA_FOURCC` | `YUYV` | `YUYV`、`MJPG`、`AUTO` 其一 |
| `WEBCAM_WIDTH` | `640` | 擷取寬度 |
| `WEBCAM_HEIGHT` | `480` | 擷取高度 |
| `WEBCAM_FPS` | `15` | 擷取幀率 |
| `WEBCAM_DETECTION_INTERVAL` | `5.0` | 辨識間隔（秒） |
| `CAMERA_INFERENCE_IMGSZ` | `640` | 攝影機推理圖片尺寸（可設較小值加速） |
| `CAMERA_DEBUG` | `false` | 印出詳細攝影機記錄 |

### 電子秤

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `SCALE_READER_MODE` | `serial` | `serial` 或 `mock` |
| `SERIAL_PORT` | `/dev/ttyUSB0` | 串列裝置路徑 |
| `SERIAL_BAUDRATE` | `9600` | 鮑率 |
| `SERIAL_TIMEOUT` | `0.1` | 每次 readline() 等待秒數 |
| `SERIAL_READ_RETRIES` | `5` | 每次讀取最多嘗試行數 |
| `SCALE_BG_POLL_INTERVAL` | `0.1` | 背景輪詢間隔（秒，10 Hz） |
| `SCALE_DEBUG` | `false` | 印出每次原始讀值與過濾結果 |

### Web 伺服器

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `BACKEND_HOST` | `0.0.0.0` | 監聽位址 |
| `BACKEND_PORT` | `5000` | 監聽埠號 |

### 覆蓋範例

```bash
# 無電子秤（mock 模式）
SCALE_READER_MODE=mock ./run_jetson.sh

# 電子秤接在 /dev/ttyACM0
SERIAL_PORT=/dev/ttyACM0 ./run_jetson.sh

# 退回 PyTorch 推理（較慢）
DETECTOR_BACKEND=pt ./run_jetson.sh

# 攝影機接在 /dev/video1
CAMERA_SOURCE=/dev/video1 ./run_jetson.sh

# 縮小推理圖片尺寸（降低延遲）
CAMERA_INFERENCE_IMGSZ=416 ./run_jetson.sh

# 啟用所有偵錯記錄
CAMERA_DEBUG=true SCALE_DEBUG=true ./run_jetson.sh

# 指定其他埠號
BACKEND_PORT=8080 ./run_jetson.sh
```

---

## 12. 啟動應用程式

```bash
cd /home/camlion/projects/instrument
source venv/bin/activate
./run_jetson.sh
```

### 預期的啟動輸出

```
========================================================
 Jetson Startup Diagnostics
========================================================
  Python     : 3.8.x
  Arch       : aarch64
  PyTorch    : 2.4.1  (CUDA=no)
  OpenCV     : 4.x.x
  ultralytics: 8.4.52
  Flask      : 3.0.3
  Model      : models/best.pt  (OK)
  Camera src : /dev/video0
  Video devs : /dev/video0
  Capture    : 640x480 @ 15 fps  fourcc=YUYV
  Detect int : 5 s
  Detector   : backend=onnx
  ONNX model : models/best.onnx  (OK)
  onnxruntime: 1.19.2  providers=[..., CPUExecutionProvider]
  Scale mode : serial
  Scale port : /dev/ttyUSB0  (baud=9600)
               -> port found OK
========================================================
...
 * Running on http://0.0.0.0:5000
 * Running on http://192.168.0.xxx:5000
```

若出現 `[WARNING] /dev/ttyUSB0 not found!`，代表電子秤未接上或權限不足。  
若出現 `[WARNING] /dev/video0 not found!`，代表攝影機未接上。  
兩者皆不影響 Flask 啟動，但相關功能無法使用。

### 停止應用程式

```
Ctrl+C
```

---

## 13. 安全關機按鈕設定

Web UI 右上角有一個「⏻ 安全關機」按鈕，點擊後會停止所有服務（攝影機、辨識、電子秤）並關閉 Jetson 系統電源。

> **預設停用。** 需完成 sudoers 設定後才能啟用，否則按鈕會回傳 403 錯誤，不會執行任何操作。

### A. 確認環境變數設定

`run_jetson.sh` 中的預設值為：

```bash
export ENABLE_SYSTEM_SHUTDOWN="${ENABLE_SYSTEM_SHUTDOWN:-false}"
```

啟動時若不設定，關機按鈕功能停用。若要啟用，以下列方式執行：

```bash
ENABLE_SYSTEM_SHUTDOWN=true ./run_jetson.sh
```

### B. 設定 sudoers（必要步驟）

Flask 應用程式以一般使用者身份執行（例如 `camlion`），必須允許該使用者免密碼執行 `shutdown`。

**步驟 1 — 以 root 身份建立 sudoers 規則：**

```bash
sudo visudo -f /etc/sudoers.d/instrument-shutdown
```

> 若系統未安裝 `nano`，`visudo` 預設開啟 `vi`。vi 操作方式：按 `i` 進入插入模式，輸入內容後按 `Esc`，再輸入 `:wq` 儲存離開。
> 若偏好 nano，先安裝：`sudo apt install -y nano`

**步驟 2 — 在編輯器中輸入以下內容（將 `camlion` 改為實際使用者名稱）：**

```
camlion ALL=(root) NOPASSWD: /sbin/shutdown, /sbin/poweroff, /usr/sbin/shutdown, /usr/sbin/poweroff
```

**步驟 3 — 設定正確的檔案權限：**

```bash
sudo chmod 440 /etc/sudoers.d/instrument-shutdown
```

**步驟 4 — 驗證語法：**

```bash
sudo visudo -c -f /etc/sudoers.d/instrument-shutdown
```

輸出應為 `/etc/sudoers.d/instrument-shutdown: parsed OK`。

**步驟 5 — 測試免密碼關機（先不要真的關機）：**

```bash
sudo -n /sbin/shutdown --help 2>&1 | head -1
```

若不出現密碼提示，代表 sudoers 設定正確。

### C. 啟用並測試

```bash
ENABLE_SYSTEM_SHUTDOWN=true ./run_jetson.sh
```

開啟 Web UI 後點選右上角「⏻ 安全關機」按鈕，確認出現確認對話框。確認後系統將在約 0.5 秒後執行 `sudo shutdown -h now`。

### D. 安全注意事項

- 若 `ENABLE_SYSTEM_SHUTDOWN=false`（預設），按鈕點選後回傳 HTTP 403，不執行任何操作。
- sudoers 規則僅允許 `shutdown` 和 `poweroff`，不開放其他 root 指令。
- Kiosk 環境建議啟用此功能以便操作人員無鍵盤關機；一般開發環境保持停用。

---

## 14. 瀏覽器操作說明

在同一網路的任何電腦或 Jetson 本機開啟瀏覽器，輸入：

```
http://<Jetson IP>:5000
```

例如：`http://192.168.0.234:5000`

### 操作流程

| 動作 | 說明 |
|------|------|
| **頁面載入** | 僅啟動 `/status` 每 2 秒輪詢；不開始重量輪詢 |
| **◉ 開啟攝影機** | 開啟 `/dev/video0`，顯示 MJPEG 即時預覽；不啟動辨識 |
| **⚐ 開始辨識**（攝影機模式）| 每 5 秒執行 ONNX 推理；啟動背景電子秤輪詢；`/api/weight` 開始輪詢 |
| **⏹ 停止辨識** | 停止推理；停止 `/api/weight` 輪詢；攝影機預覽保持開啟 |
| **⏹ 關閉攝影機** | 釋放 `/dev/video0`；若辨識中則一併停止辨識 |
| **↑ 上傳圖片** | 不需開啟攝影機；拖放或選擇圖片後點 ⚐ 開始辨識 |

### 標準重量

- 盤點統計中的「標準重量」由前端依目前標準數量 × `class_weight.json` 中的單位重量計算。
- 手動修改標準數量後，標準重量立即更新，**不受 `/status` 輪詢覆蓋**。

### Flask 記錄說明

- Flask 記錄中每約 2 秒出現一筆 `/status?client=...` 是**正常現象**，代表前端輪詢正常運作，**不是錯誤，不需要處理**。
- `/api/weight?client=...` **只應在辨識進行中出現**。若辨識停止後仍持續出現，請查看第 18 節 F 項。

### 多分頁注意事項

- 每個瀏覽器分頁會開啟獨立的 MJPEG 串流連線。
- 辨識與重量輪詢各自獨立運作，不同分頁不互相干擾。
- 若發現 `/api/weight` 在停止辨識後仍持續出現在 Flask 記錄中，請確認是否有其他分頁開著。

---

## 15. Kiosk / 觸控螢幕模式

若使用 7 吋 HDMI 顯示器作為固定顯示裝置，可用 Kiosk 模式全螢幕顯示 UI：

```bash
chromium-browser \
  --kiosk \
  --app=http://127.0.0.1:5000 \
  --force-device-scale-factor=1
```

或使用 Firefox：

```bash
firefox --kiosk http://127.0.0.1:5000
```

> **建議**：先啟動 `./run_jetson.sh` 後再開啟瀏覽器，確保 Flask 已就緒。

若要開機自動啟動，可將啟動腳本加入 `~/.config/autostart/` 或使用 `systemd` 服務。

---

## 16. 開機自動啟動與 Kiosk 全螢幕

本節說明如何讓 Flask 伺服器在開機後自動啟動，並在桌面登入後自動開啟 Chromium Kiosk 全螢幕模式。所有腳本已放置於 `scripts/` 和 `deploy/` 目錄中。

### A. 安裝 Flask 伺服器自動啟動（systemd）

```bash
cd /home/camlion/projects/instrument
sudo scripts/install_autostart_server.sh
```

此腳本會：

1. 將 `deploy/systemd/instrument-server.service` 複製到 `/etc/systemd/system/`
2. 執行 `systemctl daemon-reload`
3. 啟用並重啟 `instrument-server.service`（含 `ENABLE_SYSTEM_SHUTDOWN=true`）

**安裝後驗證**：

```bash
# 確認服務為 active (running)
systemctl status instrument-server.service --no-pager

# 確認 Flask 監聽 0.0.0.0:5000
ss -ltnp | grep 5000 || true

# 查看啟動記錄（應有 ONNX 載入、Scale port、Shutdown ENABLED）
journalctl -u instrument-server.service -n 100 --no-pager
```

啟動記錄預期包含：

```
  onnxruntime: <版本>  providers=[CPUExecutionProvider]
  Scale port : /dev/ttyUSB0  (baud=9600)
               -> port found OK
  Shutdown   : ENABLED
```

### B. 查看伺服器日誌

```bash
# 目前狀態
systemctl status instrument-server.service --no-pager

# 即時追蹤日誌
journalctl -u instrument-server.service -f

# 查看最近 100 行
journalctl -u instrument-server.service -n 100 --no-pager
```

### C. 安裝 Kiosk 桌面自動啟動

```bash
scripts/install_kiosk_autostart.sh
```

此腳本會：

1. 將 `deploy/autostart/instrument-kiosk.desktop` 複製到 `~/.config/autostart/`
2. 建立 Chromium 專用設定目錄 `~/.config/instrument-chromium`
3. 設定正確的擁有者和權限

**安裝後驗證**：

```bash
ls -l /home/camlion/.config/autostart/instrument-kiosk.desktop
cat /home/camlion/.config/autostart/instrument-kiosk.desktop
```

預期 `.desktop` 內容：

```
[Desktop Entry]
Type=Application
Name=Instrument Kiosk
Comment=Launch Instrument Recognition Kiosk
Exec=/home/camlion/projects/instrument/scripts/start_kiosk.sh
Terminal=false
X-GNOME-Autostart-enabled=true
```

> **注意**：`~/.config/autostart/` 只在使用者完成圖形桌面登入後才會執行。若要完全無人值守開機，需在顯示管理員（LightDM）中啟用 `camlion` 自動登入，詳見下方 J 項。

`start_kiosk.sh` 啟動時會等待 Flask 伺服器在 `http://127.0.0.1:5000` 就緒（最多 120 秒）後再開啟 Chromium，確保頁面不會在伺服器啟動前載入。

### D. 停用螢幕休眠與螢幕保護程式

在桌面 session 中執行：

```bash
scripts/disable_sleep_blanking.sh
```

此腳本執行以下指令（失敗自動忽略）：

```bash
xset s off
xset s noblank
xset -dpms
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.desktop.screensaver lock-enabled false
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
```

若要永久套用，可選擇性編輯 `/etc/systemd/logind.conf`（需 root）：

```
HandleLidSwitch=ignore
IdleAction=ignore
IdleActionSec=0
```

然後執行 `sudo systemctl restart systemd-logind`。

### E. 重新開機測試

```bash
sudo reboot
```

重開機後預期行為：

| 步驟 | 時序 | 預期 |
|------|------|------|
| 1 | 開機完成 | `camlion` 自動登入（LightDM） |
| 2 | systemd 啟動 | `instrument-server.service` 自動啟動，Flask 監聽 0.0.0.0:5000 |
| 3 | 桌面 session 啟動 | `~/.config/autostart/instrument-kiosk.desktop` 觸發 `start_kiosk.sh` |
| 4 | 等待伺服器 | `start_kiosk.sh` 以 `curl` 輪詢 `http://127.0.0.1:5000`，最多等 120 秒 |
| 5 | 伺服器就緒 | Chromium 以 Kiosk 全螢幕模式開啟，顯示 Web UI |
| 6 | 操作中 | 螢幕不休眠、不出現螢幕保護程式；⏻ 安全關機按鈕可用 |

啟動記錄診斷（`run_jetson.sh` 輸出）確認項目：

- ONNX Runtime 已載入（`providers=[CPUExecutionProvider]`）
- 攝影機路徑存在（`/dev/video0`）
- Scale port 找到（`port found OK`）
- `Shutdown   : ENABLED`（`ENABLE_SYSTEM_SHUTDOWN=true` 時）

### F. 排錯

**Kiosk 顯示「waiting for Flask server on http://127.0.0.1:5000」且持續等待：**

先確認 Flask 是否確實監聽：
```bash
ss -ltnp | grep 5000 || true
```

若 Flask 有在監聽但 Kiosk 仍在等待，確認 `curl` 是否安裝：
```bash
which curl
curl -I http://127.0.0.1:5000
```

若 `curl` 未安裝：
```bash
sudo apt install -y curl
```

---

**`curl: command not found`**

原因：`curl` 未安裝（§5 必要套件）。

修復：
```bash
sudo apt install -y curl
```

---

**Chromium 未開啟（伺服器正在執行）：**

```bash
# 確認自動登入設定
cat /etc/lightdm/lightdm.conf

# 確認 .desktop 檔案存在
ls -l /home/camlion/.config/autostart/instrument-kiosk.desktop
cat /home/camlion/.config/autostart/instrument-kiosk.desktop

# 確認 Chromium 是否正在執行
ps aux | grep -i chromium
ps aux | grep -i start_kiosk
```

最常見原因：LightDM 自動登入未設定（見 J 項），`~/.config/autostart/` 從未執行。

**手動測試 Kiosk（在有桌面 session 的情況下）：**
```bash
DISPLAY=:0 /home/camlion/projects/instrument/scripts/start_kiosk.sh
```

---

**伺服器未啟動：**
```bash
systemctl status instrument-server.service
journalctl -u instrument-server.service -n 100 --no-pager
```
常見原因：venv 路徑錯誤、模型檔案不存在、串列埠被佔用。

**手動測試伺服器腳本（Ctrl+C 停止）：**
```bash
./scripts/start_instrument_server.sh
```

---

**編輯設定檔時 `nano: command not found`：**

使用 `vi` 代替：
```bash
sudo vi /etc/lightdm/lightdm.conf
# 操作：i 進入插入模式，Esc 離開插入，:wq 儲存
```

或先安裝 nano：
```bash
sudo apt install -y nano
```

或使用本指南中的 `sudo tee` 指令，無需文字編輯器。

---

**Chrome 出現「上次未正常關閉」泡泡：**
- `start_kiosk.sh` 已加入 `--disable-session-crashed-bubble` 和 `--disable-infobars` 旗標。

---

**螢幕仍然休眠：**
```bash
xset q   # 查看目前 DPMS / screensaver 狀態
```
再次執行 `scripts/disable_sleep_blanking.sh` 或設定 LightDM 的 `xserver-command` 加入 `-s 0 -dpms`。

### G. 停止 / 停用自動啟動

停用 systemd 伺服器服務：

```bash
sudo systemctl stop instrument-server.service
sudo systemctl disable instrument-server.service
```

移除 Kiosk 自動啟動：

```bash
rm ~/.config/autostart/instrument-kiosk.desktop
```

### H. 手動測試腳本

在安裝前可先手動測試各腳本：

```bash
# 測試伺服器啟動腳本（Ctrl+C 停止）
./scripts/start_instrument_server.sh

# 測試 Kiosk 腳本（需在桌面 session 中，手動關閉 Chromium）
./scripts/start_kiosk.sh

# 測試停用螢幕休眠
./scripts/disable_sleep_blanking.sh
```

### I. 完整一次性安裝指令

所有元件設定完成後，可依序執行以下完整流程：

```bash
cd /home/camlion/projects/instrument

# 1. 安裝 systemd 服務
sudo scripts/install_autostart_server.sh

# 2. 安裝 Kiosk 桌面自動啟動
scripts/install_kiosk_autostart.sh

# 3. 停用螢幕休眠（在桌面 session 中執行）
scripts/disable_sleep_blanking.sh

# 4. 設定 LightDM 自動登入（備份現有設定後覆寫）
sudo cp /etc/lightdm/lightdm.conf /etc/lightdm/lightdm.conf.bak 2>/dev/null || true
sudo tee /etc/lightdm/lightdm.conf >/dev/null <<'EOF'
[Seat:*]
autologin-user=camlion
autologin-user-timeout=0
EOF

# 5. 設定安全關機 sudoers（見 §13）
sudo visudo -f /etc/sudoers.d/instrument-shutdown
sudo chmod 440 /etc/sudoers.d/instrument-shutdown
sudo visudo -c -f /etc/sudoers.d/instrument-shutdown

# 6. 重開機
sudo reboot
```

### J. 啟用桌面自動登入（LightDM）

> **重要**：若未設定自動登入，Chromium Kiosk 在重開機後不會自動啟動（因為 `~/.config/autostart/` 僅在圖形 session 啟動後執行）。若只需要 Flask 伺服器自動啟動（無 Kiosk），可略過此步驟。

若使用 LightDM（Jetson Nano Ubuntu 18.04 預設）：

```bash
# 備份現有設定
sudo cp /etc/lightdm/lightdm.conf /etc/lightdm/lightdm.conf.bak 2>/dev/null || true

# 寫入自動登入設定（推薦方式，不需要 nano）
sudo tee /etc/lightdm/lightdm.conf >/dev/null <<'EOF'
[Seat:*]
autologin-user=camlion
autologin-user-timeout=0
EOF
```

或使用文字編輯器手動修改：

```bash
sudo nano /etc/lightdm/lightdm.conf
# 若未安裝 nano：sudo apt install -y nano
# 或使用 vi：sudo vi /etc/lightdm/lightdm.conf
```

**驗證設定**：

```bash
cat /etc/lightdm/lightdm.conf
```

預期輸出：

```
[Seat:*]
autologin-user=camlion
autologin-user-timeout=0
```

重開機後即自動登入 `camlion`，Kiosk 隨之啟動。

---

## 17. 驗證清單

完成部署後，逐一確認以下項目：

- [ ] **venv Python 版本正確**：`which python3` 指向 venv，`python3 --version` 顯示 3.8.x
- [ ] **所有套件 import 成功**：執行第 7 節的完整 import 測試，MISSING_PACKAGES=(none)
- [ ] **模型檔案存在**：`best.pt`、`best.onnx`、`class_weight.json` 皆存在
- [ ] **攝影機預覽測試通過**：`python3 tools/test_camera_preview.py` 無錯誤
- [ ] **串列埠權限正確**：`/dev/ttyUSB0 rw=True`（需在加入 dialout 群組並重新登入後）
- [ ] **上傳圖片辨識成功**：上傳一張圖片，點開始辨識，結果表格出現器械清單
- [ ] **攝影機開啟 / 停止 / 重新開啟**：三次操作均正常，無卡住或殘留串流
- [ ] **開始辨識更新結果**：盤點結果表格每 5 秒更新一次
- [ ] **重量顯示更新**：實際重量在辨識中約每 0.5 秒更新
- [ ] **停止辨識後重量輪詢停止**：Flask 記錄中 `/api/weight` 不再出現
- [ ] **標準重量由 class_weight.json 計算**：盤點統計的標準重量與 JSON 中設定一致
- [ ] **`./run_jetson.sh` 啟動無 ModuleNotFoundError**

---

## 18. 常見問題排查

### A. `source venv/bin/activate` 後仍是系統 Python 3.6

**原因**：複製的 venv 中 `activate` 腳本路徑指向舊機器使用者目錄。

**確認**：
```bash
grep VIRTUAL_ENV venv/bin/activate | head -2
```

若輸出包含舊路徑（如 `/home/ncut/...`），執行第 6 節方法 B 的修復步驟。

---

### B. `ModuleNotFoundError: No module named 'cycler'`（或 `six` / `urllib3` / `certifi` / `idna`）

**原因**：複製的 venv 缺少原始機器系統 Python 提供的套件。

**修復**：
```bash
source venv/bin/activate
pip install cycler six urllib3 certifi idna
```

安裝後將套件名稱加入 `requirements.txt`。

---

### C. 開啟 `/dev/ttyUSB0` 時 Permission denied

**確認**：
```bash
ls -l /dev/ttyUSB0
# crw-rw---- 1 root dialout ...
groups
# 輸出中應有 dialout
```

**修復**：
```bash
sudo usermod -aG dialout $USER
# 必須登出再登入
```

若需立即生效（僅當前終端有效）：
```bash
newgrp dialout
```

---

### D. 無法開啟 `/dev/video0`

**確認**：
```bash
ls -l /dev/video*
fuser -v /dev/video0 2>/dev/null || echo "(無其他程序佔用)"
```

**可能原因與修復**：

| 原因 | 修復 |
|------|------|
| 攝影機未接上 | 確認 USB 連接，執行 `lsusb` 查看裝置 |
| 裝置路徑不同 | 嘗試 `CAMERA_SOURCE=/dev/video1 ./run_jetson.sh` |
| 其他程序佔用 | 關閉佔用程序後再啟動 |

---

### E. 攝影機串流重複或延遲

**確認**：觀察 Flask 記錄，是否有多個 `/camera/stream?session=...` 的連線。

**修復**：
- 關閉多餘的瀏覽器分頁，只保留一個分頁
- 確認記錄中 `client=` 參數是否相同（相同代表同一分頁重複連線）

---

### F. 停止辨識後 `/api/weight` 仍持續出現在記錄中

**確認**：檢查 Flask 記錄中 `/api/weight?client=...&reason=...` 的 `client` 參數。

| `client` | `reason` | 說明 |
|----------|----------|------|
| 相同 | `loop` | 同一分頁的輪詢仍在執行，可能是 JS 狀態未同步 |
| 不同 | 任何 | 另一個瀏覽器分頁或裝置仍在輪詢 |
| 空白 | 任何 | 程式碼中有直接呼叫 API 的路徑，請回報 |

**修復**：關閉所有多餘分頁，重新整理頁面。

---

### G. ONNX 模型缺失或損毀

**症狀**：啟動記錄顯示 `MISSING — run: python3 -c "..."`

**修復**：

1. 若 `best.pt` 存在，重新匯出：
   ```bash
   source venv/bin/activate
   python3 -c "from ultralytics import YOLO; YOLO('models/best.pt').export(format='onnx', opset=12)"
   ```

2. 或暫時切換到 PyTorch 後端：
   ```bash
   DETECTOR_BACKEND=pt ./run_jetson.sh
   ```

---

### H. Flask 啟動後首次辨識特別慢

**正常現象**：ONNX Runtime 在首次推理時需要初始化（1–5 秒）。後續推理約 1.8–2.0 秒（640×640）。若要縮短每次推理時間，可設：

```bash
CAMERA_INFERENCE_IMGSZ=416 ./run_jetson.sh
```

---

## 19. Git 注意事項

### 不應提交的檔案

以下由 `.gitignore` 排除，**請勿手動 `git add`**：

```
tmp/
models/best.pt
models/best.onnx
models/*.engine
venv/
__pycache__/
*.pyc
```

### 應提交的檔案

```
models/class_weight.json      # 器械單位重量，影響標準重量計算
requirements.txt              # 依賴清單，每次新增套件後更新
output/standards.json         # 可選：若此為預設標準配置則提交
```

### output/ 目錄說明

| 檔案 | 說明 |
|------|------|
| `output/standards.json` | 每類器械的標準數量（執行時可修改）；若要作為新機器的預設值則提交 |
| `output/unit_weights.json` | 舊版每類重量（現已由 `class_weight.json` 取代）；可提交作為記錄，但非計算依據 |

### 查看目前修改

```bash
git status
git diff -- requirements.txt
git log --oneline -5
```

---

## 20. 最終快速驗證指令

完成所有設定後，依序執行以下指令作為最終確認：

```bash
cd /home/camlion/projects/instrument
source venv/bin/activate

# 1. 套件 import 全部通過
python3 -c "
import flask, cv2, numpy, ultralytics, onnxruntime, serial, torch
import matplotlib, cycler, six, urllib3, certifi, idna, requests
print('所有套件 import 通過')
"

# 2. 模型載入確認
python3 - <<'PY'
from ultralytics import YOLO
model = YOLO("models/best.pt")
print(f"模型載入成功 — 類別數：{len(model.names)}")
PY

# 3. 攝影機功能測試
python3 tools/test_camera_preview.py

# 4. 串列埠權限確認
python3 - <<'PY'
import os
p = "/dev/ttyUSB0"
print(f"{p}  exists={os.path.exists(p)}  rw={os.access(p, os.R_OK | os.W_OK)}")
PY

# 5. 啟動應用程式（確認後按 Ctrl+C 停止）
./run_jetson.sh
```

**啟動成功標準**：
- Flask 在 `:5000` 上線
- 記錄顯示 `Using ONNX Runtime ... with CPUExecutionProvider`
- 無 `ModuleNotFoundError`
- 無 `[MISSING]` 型模型警告

---

*最後更新：2026-05-26 | 分支：jetson-gpu-experiment*
