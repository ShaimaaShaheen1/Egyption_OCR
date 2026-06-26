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
   └── Models/
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





# ID Card OCR API - Linux Deployment Guide

This guide explains how to deploy the ID Card OCR API using Docker on a Linux machine. 

Unlike Windows, Linux requires the **NVIDIA Container Toolkit** to allow Docker to communicate with your GPU.

---

## 1. Prerequisites & GPU Setup

* **Verify NVIDIA Driver:** Ensure your host machine has the NVIDIA driver installed. Run this in your terminal:
  ```bash
  nvidia-smi
  ```
  *(You should see your GPU and CUDA version. If not, install the NVIDIA driver first).*

* **Install Docker Engine:** Ensure Docker is installed and running on your system.

* **Install NVIDIA Container Toolkit:** Plain Docker cannot see the GPU by itself. Run the following commands to install the toolkit (for Ubuntu/Debian-based systems):
  ```bash
  curl -fsSL [https://nvidia.github.io/libnvidia-container/gpgkey](https://nvidia.github.io/libnvidia-container/gpgkey) | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  
  curl -s -L [https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list](https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list) | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

* **Verify GPU Access in Docker:**
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
  ```
  *(If this prints your GPU info, Docker is successfully configured to use the GPU).*

---

## 2. Project Files & The `models` Folder (IMPORTANT)

**⚠️ LINUX CASE SENSITIVITY WARNING:** Unlike Windows, Linux is strictly case-sensitive. Ensure your folder names exactly match the paths written in your Python code (e.g., `models` must be exactly lowercase if the code expects it).

To avoid `FileNotFoundError`, structure your directory exactly as shown below:

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
    └── TROCRcheckpoint_300_final_version/
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

Navigate to the project folder and run the following commands:

```bash
cd ~/id-card-ocr
docker compose build
docker compose up -d
```

Check if the server started cleanly:

```bash
docker compose logs -f
```

You should see `Using device: cuda` and `Model converted to Half Precision (FP16)` confirming that the GPU and FP16 are active.

---

## 5. Test the API

* **API Documentation (Swagger UI):** Run `curl http://localhost:8000/docs` to verify the API is serving.
* **Web Form Interface:** Open `http://<server-ip>:8000/` in a web browser to access the upload form, and enter the API key/secret from `credentials.json` when prompted.

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

* **No NVIDIA GPU available? (CPU Fallback):** If the target Linux machine doesn't have a GPU, open `docker-compose.yml`, **delete the `deploy` section** (containing GPU reservations), and run `docker compose up -d`. The models will automatically run on the CPU.
* **`could not select device driver "" with capabilities: [[gpu]]`**: The NVIDIA Container Toolkit isn't installed or configured correctly. Redo the steps in Section 1.
* **Container runs but stays on CPU**: Check `nvidia-smi` on the host to ensure no other process is hogging all GPU memory.
* **`libGL.so.1: cannot open shared object file`**: This is already handled by the Dockerfile. If you see it anyway, rebuild your container cleanly using `docker compose build --no-cache`.
