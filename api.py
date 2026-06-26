import os
import json
import uuid
import time
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_403_FORBIDDEN, HTTP_401_UNAUTHORIZED, HTTP_429_TOO_MANY_REQUESTS
import uvicorn
from pydantic import BaseModel, Field
from OCR import func
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api_access.log'
)
logger = logging.getLogger("id_card_api")

app = FastAPI(title="ID Card API", description="API for ID card recognition")

# Host/port are read from the environment so they can be set per-deployment
# (e.g. via docker-compose / docker run -e) without touching the code.
# 0.0.0.0 is required inside Docker so the port is reachable from outside
# the container; 127.0.0.1 would only accept connections from inside it.
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))



# Path to credentials file
CREDENTIALS_FILE = "credentials.json"

# Rate limiting settings
RATE_LIMIT_REQUESTS = 100  # Maximum requests per time window
RATE_LIMIT_WINDOW = 3600   # Time window in seconds (1 hour)
request_history = {}

# Load API credentials from JSON file
def load_credentials() -> dict:
    try:
        if not os.path.exists(CREDENTIALS_FILE):
            logger.error(f"Credentials file {CREDENTIALS_FILE} not found")
            raise FileNotFoundError(f"Credentials file {CREDENTIALS_FILE} not found")
        
        with open(CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)
            
        # Validate credentials structure
        if not all(key in credentials for key in ["api_key", "api_secret"]):
            logger.error("Invalid credentials format")
            raise ValueError("Invalid credentials format")
            
        return credentials
    except json.JSONDecodeError:
        logger.error("Invalid JSON in credentials file")
        raise ValueError("Invalid JSON in credentials file")

# Rate limiting check
def check_rate_limit() -> bool:
    current_time = time.time()
    client_id = "global"  # Using a global rate limit instead of per-client
    
    # Initialize history if not exists
    if client_id not in request_history:
        request_history[client_id] = []
    
    # Clean old requests
    request_history[client_id] = [
        timestamp for timestamp in request_history[client_id] 
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if rate limit exceeded
    if len(request_history[client_id]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit exceeded")
        return False
    
    # Add current request timestamp
    request_history[client_id].append(current_time)
    return True

# Authentication dependency
async def verify_api_key(
    x_api_key: str = Header(..., description="API Key for authentication"),
    x_api_secret: str = Header(..., description="API Secret for authentication")
):
    # Log access attempt
    logger.info(f"Authentication attempt")
    
    try:
        # Check rate limit
        if not check_rate_limit():
            logger.warning(f"Rate limit exceeded")
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Try again later."
            )
        
        # Load and verify credentials
        credentials = load_credentials()
        valid_key = credentials.get("api_key")
        valid_secret = credentials.get("api_secret")
        
        # Check if credentials exist
        if valid_key is None or valid_secret is None:
            logger.error("Invalid credentials configuration")
            raise HTTPException(
                status_code=HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials configuration"
            )
        
        # Constant-time comparison to prevent timing attacks
        key_valid = secrets.compare_digest(x_api_key, valid_key)
        secret_valid = secrets.compare_digest(x_api_secret, valid_secret)
        
        if not (key_valid and secret_valid):
            logger.warning(f"Authentication failed")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid API credentials"
            )
        
        logger.info(f"Authentication successful")
        return {"authenticated": True}
        
    except FileNotFoundError as e:
        logger.error(f"Credentials file error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"Credential validation error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

class IDResponse(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    accuracy: float
    id_data: Dict[str, Any]

# Add a route to serve your HTML form
from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
async def serve_form():
    """Serve the HTML form for ID card recognition"""
    return FileResponse("form.html")


@app.post("/api/id-recognition", response_model=IDResponse)
async def process_id_card(
    file: UploadFile = File(...),
    auth_info: dict = Depends(verify_api_key)
):
    """
    Process ID card image and return recognition results
    """
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            # Process the image using the temporary file path
            ocr_result = func(temp_file_path)
            
            # Ensure the OCR result is properly formatted
            if isinstance(ocr_result, dict):
                response_data = ocr_result
            else:
                response_data = {
                    "first_name":ocr_result[1],
                    "id":ocr_result[0],
                    "final_name":ocr_result[2]

                }
            
            # Format the response according to IDResponse model
            response = {
                "timestamp": time.time(),
                "accuracy": 0.95,  # You might want to get this from your OCR function
                "id_data": response_data
            }
            
            logger.info("Processing successful")
            return response
        finally:
            os.unlink(temp_file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"}
    )

# Run the application directly if this file is executed
def run_app():
    print("Starting OCR API server...")
    print(f"API documentation available at:")
    print(f"  - Swagger UI: http://{APP_HOST}:{APP_PORT}/docs")
    print(f"Access the form at http://{APP_HOST}:{APP_PORT}/")
    
    # Import here to avoid errors if uvicorn is not installed
    import uvicorn
    
    # Run the server
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)

# This allows you to run the file directly
if __name__ == "__main__":
    run_app()
