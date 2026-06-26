# ID Card OCR API - Deployment Guide

This guide explains how to deploy the ID Card OCR API using Docker. 

Windows + Docker Desktop handles GPU passthrough itself via WSL2, so no separate "NVIDIA Container Toolkit" install is needed on Windows. For Linux, you must install the NVIDIA Container Toolkit first.

---

## 1. Prerequisites

* **NVIDIA GPU:** Install the latest "Game Ready" or "Studio" driver from the official NVIDIA website.
* **Docker Desktop (Windows):** Open Docker Desktop → **Settings** → **General** → confirm **"Use the WSL 2 based engine"** is checked.
* **Verify GPU Access:** Open your terminal (or PowerShell) and run this command to ensure Docker can see your GPU:
  ```bash
  docker run --rm --gpus=all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
  If this prints your GPU's name and CUDA version, GPU passthrough is working.

---

## 2. Project Files & The `models` Folder (IMPORTANT)

To avoid any path errors (`FileNotFoundError`), your project directory must be structured exactly as shown below.

1. Download the main project files (including `Dockerfile`, `docker-compose.yml`, `api.py`, `OCR.py`, etc.).
2. **Download the `models` folder** and place it directly inside the main project directory.
3. Ensure your folder structure looks exactly like this:

   ```text
   id-card-ocr/
   ├── docker-compose.yml
   ├── api.py
   ├── OCR.py
   ├── detectron.py
   ├── requirements.txt
   ├── api_access.log
   ├── Dockerfile
   ├── .dockerignore
   ├── form.html
   └── models/
       ├── arabic_digit_model3.pth
       ├── TROCRcheckpoint_300_final_version/
       └── ....
  
   ```

---

## 3. Add `credentials.json`

This file is **not** included in the image for security. Create a file named `credentials.json` in the project root (the same folder as `docker-compose.yml`) with the following exact content:

```json
{
  "api_key": "your_real_api_key",
  "api_secret": "your_real_api_secret"
}
```

---

## 4. Build and Run

Open your terminal, navigate to the main project folder, and run the following commands:

```bash
docker compose build
docker compose up -d
```

Check if the server started cleanly:

```bash
docker compose logs -f
```

You should see `Using device: cuda` and `Model converted to Half Precision (FP16)` lines confirming that the GPU and FP16 are active.

---

## 5. Test the API

* **API Documentation (Swagger UI):** Open `http://localhost:8000/docs` in a web browser.
* **Web Form Interface:** Open `http://localhost:8000/` to access the upload form, and enter the API key/secret from `credentials.json` when prompted.

---

## 6. Common Commands

```bash
docker compose down               # Stop and remove the container
docker compose up -d              # Start the container in the background
docker compose logs -f            # View live server logs
docker compose build --no-cache   # Force a clean rebuild of the image
```

---

## 7. Troubleshooting

* **No NVIDIA GPU available? (CPU Fallback):** If you are running this on a laptop or system without an NVIDIA GPU, open `docker-compose.yml`, **delete the `deploy` section** (which contains the GPU reservations), and run `docker compose up -d`. The models will automatically run on your CPU instead.
* **`could not select device driver "" with capabilities: [[gpu]]`**: Docker isn't configured for GPU access. On Windows, recheck WSL2 settings; on Linux, ensure the NVIDIA Container Toolkit is installed.
* **Container runs but stays on CPU**: Confirm step 1's test command works. If it doesn't, the issue is at the environment level, not the app.
* **Build is very slow / runs out of disk space**: The base CUDA+PyTorch image plus the model weights add up to several GB. Make sure Docker Desktop's disk image size limit (Settings → Resources → Advanced) is large enough.
