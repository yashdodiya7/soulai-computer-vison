# 🚀 CIFAR-10 Image Classification Model (Google Colab Setup)

This repository contains a **deep learning model** trained on the **CIFAR-10 dataset** using **ResNet9**. You can easily train the model in **Google Colab** without needing a local setup.

---

## 🐳 Quick Docker Setup
The API is available as a **Docker image** on Docker Hub. You can pull and run it for instant access on localhost.

### 1️⃣ Pull the Docker Image
```sh
docker pull devyashdodiya/fastapi-app
```

### 2️⃣ Run the API Container
```sh
docker run -p 8000:8000 --env API_USERNAME=admin --env API_PASSWORD=admin devyashdodiya/fastapi-app
````
---

## 🔥 Train the Model in Google Colab

### 1️⃣ Open Google Colab
 - Add **`model.py`** file code to the New Colab Notebook

---

### 2️⃣ Change Runtime to **GPU (T4)**
1. In Colab, go to **Runtime** → **Change runtime type**.
2. Select **GPU** and ensure it’s set to **T4** for better performance.
3. Click **Save**.

✅ This will speed up training significantly!

---

### 3️⃣ Run All Cells
- Click **Runtime** → **Run all** or press `Ctrl + F9`.
- The notebook will:
  ✅ Install all required **dependencies** automatically.  
  ✅ Download the **CIFAR-10 dataset**.  
  ✅ Train the **ResNet9** model.  
  ✅ Save the trained model as **`best_cifar10_model.pt`** in the **root directory**.  

---

## 📥 Download the Trained Model
- After training completes, download the model to your local machine


# 🚀 CIFAR-10 Image Classification API (FastAPI)

This API provides an endpoint for classifying images into **CIFAR-10 categories** using a pre-trained **ResNet9 model**. The API is built with **FastAPI** and secured with **Basic Authentication**.

## 📂 Project Structure
```
📦 api-project
├── 📁 app/                    # Application directory
│   ├── app.py                 # FastAPI service for image classification
│   ├── best_cifar10_model.pt  # Trained model for image classification
├── requirements.txt           # Python dependencies
|── model.py                   # ResNet9 model for CIFAR-10 classification
├── .env                       # Environment variables for authentication
└── README.md
```

## ⚙️ Setup & Installation

### 1️⃣ Create a Virtual Environment
First, create and activate a virtual environment:

**Windows (cmd/PowerShell):**
```sh
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux (bash/zsh):**
```sh
python -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Dependencies
Once the virtual environment is activated, install all required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file in the root directory with the following content:
```
API_USERNAME="admin"
API_PASSWORD="admin"
```
This sets up **Basic Authentication** for the API.

### 4️⃣ Ensure the Model File is Available
Make sure the trained model **`best_cifar10_model.pt`** is in the correct directory. Update `api.py` to load it:
```python
import torch
from model import ResNet9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "best_cifar10_model.pt")  # Update this name if needed
model = ResNet9(3, 10)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
```

### 5️⃣ Run the API Server
Start the FastAPI server using **Uvicorn** in root directory:
```sh
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

## 🖼️ Making Predictions
To classify an image, use the **`/predict/`** endpoint with **Basic Authentication**.

### 🛠️ Using `curl`
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Authorization: Basic YWRtaW46YWRtaW4=' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@image.jpg'
```
🔹 **Username:** `admin`  
🔹 **Password:** `admin`  
🔹 `YWRtaW46YWRtaW4=` is the **Base64 encoding** of `admin:admin`.

### 🛠️ Using Python
```python
import requests
from requests.auth import HTTPBasicAuth

url = "http://127.0.0.1:8000/predict/"
auth = HTTPBasicAuth("admin", "admin")

with open("image.jpg", "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files, auth=auth)

print(response.json())
```

### 📜 Example API Response
```json
{
  "predicted_class": "airplane",
  "confidence": 98.76
}
```

## 🎯 Summary
✅ **Set up a virtual environment**  
✅ **Install dependencies** from `requirements.txt`  
✅ **Create a `.env` file** with API credentials  
✅ **Ensure `best_cifar10_model.pt` is in the correct path**  
✅ **Run the API using Uvicorn**  
✅ **Use `/predict/` endpoint with authentication**  
