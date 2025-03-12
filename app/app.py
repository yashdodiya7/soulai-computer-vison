import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import logging
import secrets
import os
from fastapi.middleware.cors import CORSMiddleware

# Configure logging for API
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image_classifier_api")

# Initialize FastAPI application
app = FastAPI()

# Enable CORS for all origins (adjust for security in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic authentication setup
security = HTTPBasic()

# Load API credentials from environment variables (default: admin/admin)
VALID_USERNAME = os.getenv("API_USERNAME", "admin")
VALID_PASSWORD = os.getenv("API_PASSWORD", "admin")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify username and password"""
    is_username_correct = secrets.compare_digest(credentials.username, VALID_USERNAME)
    is_password_correct = secrets.compare_digest(credentials.password, VALID_PASSWORD)

    logger.warning(f"username: {is_username_correct}")
    logger.warning(f"password: {is_password_correct}")

    if not (is_username_correct and is_password_correct):
        logger.warning(f"Failed authentication attempt from username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    logger.info(f"Successful authentication: {credentials.username}")
    return credentials.username


# Define CIFAR-10 class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Set device for computation (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {device}")


# Define ResNet9 model architecture
class ResNet9(torch.nn.Module):
    def conv_block(self, input_channels, output_channels, use_pool=False):
        layers = [
            torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(inplace=True)
        ]
        if use_pool:
            layers.append(torch.nn.MaxPool2d(2))
        return torch.nn.Sequential(*layers)

    def __init__(self, input_channels, number_classes):
        super().__init__()

        self.conv1 = self.conv_block(input_channels, 64)
        self.conv2 = self.conv_block(64, 128, use_pool=True)
        self.residual1 = torch.nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))

        self.conv3 = self.conv_block(128, 256, use_pool=True)
        self.conv4 = self.conv_block(256, 512, use_pool=True)
        self.residual2 = torch.nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))

        self.classifier = torch.nn.Sequential(
            torch.nn.MaxPool2d(4),
            torch.nn.Flatten(),
            torch.nn.Linear(512, number_classes)
        )

    def forward(self, xb):
        layer1 = self.conv1(xb)
        layer2 = self.conv2(layer1)
        residual1 = self.residual1(layer2) + layer2
        layer3 = self.conv3(residual1)
        layer4 = self.conv4(layer3)
        residual2 = self.residual2(layer4) + layer4
        class_output = self.classifier(residual2)
        return class_output


# Load the trained model
try:
    model_path = "best_cifar10_model.pt"
    model = ResNet9(3, 10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Could not load model: {str(e)}")

# Define image transformation pipeline for preprocessing
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


# For Checking if API is running
@app.get("/")
async def root():
    return {"message": "Image Classification API is running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...), username: str = Depends(verify_credentials)):
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only image files are allowed"
        )
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(await file.read())).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        # Get predicted class and confidence score
        predicted_class = class_names[prediction.item()]
        confidence_value = confidence.item() * 100

        return {
            "predicted_class": predicted_class,
            "confidence": confidence_value
        }
    except Exception as e:
        # Log and handle errors
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
