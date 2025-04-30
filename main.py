# Install required packages:
# pip install websockets opencv-python numpy ultralytics torch

import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import logging
import torch # Keep torch import as YOLO depends on it
import socket
import sys # Import sys to handle potential exit cleanly

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Server Configuration ---
SERVER_PORT = 8765
# Listen on all available network interfaces
# You can change this to a specific IP if needed, but 0.0.0.0 is common for servers
SERVER_HOST = "0.0.0.0"

# Path to your YOLO model weights
MODEL_PATH = "weights/v9 - 64 epochs.pt"

# --- YOLO Setup ---
# Determine device for PyTorch (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using computation device: {device}")

# Load YOLO model
try:
    # Ensure the model path is correct relative to your script's location
    model = YOLO(MODEL_PATH)
    model.to(device) # Move model to the determined device
    logger.info(f"YOLO model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    logger.critical(f"Error loading YOLO model from '{MODEL_PATH}': {e}")
    logger.critical("Please ensure the model file exists and is not corrupted.")
    sys.exit(1) # Exit if the model can't be loaded

# Object classes the model is trained to detect
classNames = ["green", "red", "yellow"] # Ensure this matches your model's classes

# --- Distance Calculation (Optional, requires calibration) ---
# Placeholder constants for distance calculation.
# These values (focal_length and real_object_width) need to be calibrated
# for your specific camera and the type of traffic light you are detecting
# for the distance estimation to be accurate.
FOCAL_LENGTH_PIXELS = 650 # Example focal length in pixels (needs calibration)
REAL_TRAFFIC_LIGHT_WIDTH_CM = 20 # Example real-world average width in cm (needs verification)

def estimate_distance(focal_length, real_object_width_cm, obj_width_in_pixels):
    """
    Estimates distance to an object based on its perceived width in pixels.
    Requires camera focal length and the real-world width of the object.
    Returns distance in centimeters.
    """
    if obj_width_in_pixels <= 0:
        return float('inf') # Return infinity or a large value if object width is zero or negative
    # Formula: Distance = (Real_Object_Width * Focal_Length) / Object_Width_in_Pixels
    distance_cm = (real_object_width_cm * focal_length) / obj_width_in_pixels
    return distance_cm

# --- WebSocket Handler ---

# --- WebSocket Handler ---

# Modified signature to make 'path' optional
async def detection_handler(websocket, path=None):
    """
    WebSocket handler function to receive image data from the client,
    run YOLO detection, and send results back.

    Args:
        websocket: The WebSocket connection object.
        path: The path requested by the client (usually "/"). Default to None.
    """
    # Add a log here to explicitly see what arguments were received
    logger.info(f"Handler called with: websocket={websocket}, path={path}")
    logger.info(f"Client connected from {websocket.remote_address} on path: {path}")

    # ... rest of your handler code remains the same ...
    try:
        # Loop to receive messages from the client
        async for message in websocket:
            try:
                # Ensure the message is a string (as expected for JSON)
                if not isinstance(message, str):
                     logger.warning(f"Received non-string message type ({type(message)}) from {websocket.remote_address}")
                     await websocket.send(json.dumps({"detections": [], "error": "Server expects text messages containing JSON"}))
                     continue

                # Parse the incoming JSON message
                data = json.loads(message)

                # Extract the base64 encoded image data
                frame_data_base64 = data.get('image')

                if not frame_data_base64:
                    logger.warning(f"Received message without 'image' key or data from {websocket.remote_address}")
                    await websocket.send(json.dumps({"detections": [], "error": "No image data found in message"}))
                    continue

                # Decode base64 image data
                try:
                    img_bytes = base64.b64decode(frame_data_base64)
                    # Convert bytes to a numpy array, then to an OpenCV image
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        logger.error(f"Failed to decode image data from {websocket.remote_address}")
                        await websocket.send(json.dumps({"detections": [], "error": "Failed to decode image"}))
                        continue

                except Exception as e:
                     logger.error(f"Error during image decoding from {websocket.remote_address}: {e}", exc_info=True)
                     await websocket.send(json.dumps({"detections": [], "error": f"Image decoding failed: {e}"}))
                     continue

                # Perform object detection using YOLO
                try:
                     # Run inference. YOLO handles moving data to the device.
                     # stream=False as we process one image at a time from the client.
                     results = model(img, stream=False)

                except Exception as e:
                     logger.error(f"Error during YOLO inference for {websocket.remote_address}: {e}", exc_info=True)
                     await websocket.send(json.dumps({"detections": [], "error": f"YOLO inference failed: {e}"}))
                     continue

                # Process detection results
                detections_list = []

                for r in results:
                    boxes = r.boxes # Bounding boxes and detection details
                    # Iterate through each detected object in the frame
                    for box in boxes:
                        # Bounding box coordinates in xyxy format [x1, y1, x2, y2]
                        # Convert from tensor to list of floats
                        x1, y1, x2, y2 = box.xyxy[0].tolist()

                        # Confidence score
                        # Convert from tensor to float
                        confidence = float(box.conf[0])

                        # Class index and name
                        # Convert from tensor to int
                        cls_index = int(box.cls[0])
                        class_name = classNames[cls_index] if cls_index < len(classNames) else f"unknown_{cls_index}"

                        # Calculate object width in pixels
                        obj_width_pixels = x2 - x1

                        # Estimate distance (using placeholder constants)
                        distance_cm = estimate_distance(FOCAL_LENGTH_PIXELS, REAL_TRAFFIC_LIGHT_WIDTH_CM, obj_width_pixels)

                        # Append detection details to our list
                        detections_list.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "class": class_name,
                            "confidence": confidence,
                            "distance_cm": round(distance_cm, 2) if distance_cm != float('inf') else -1 # Add distance, round, use -1 for inf
                        })

                # logger.debug(f"Detected {len(detections_list)} objects for {websocket.remote_address}")

                # Send the detection results back to the client as JSON
                await websocket.send(json.dumps({"detections": detections_list}))

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error from {websocket.remote_address}: {e}")
                await websocket.send(json.dumps({"detections": [], "error": "Invalid JSON format received"}))
            except websockets.exceptions.ConnectionClosed:
                 # Client closed connection normally during message processing
                 logger.info(f"Client {websocket.remote_address} disconnected during message processing.")
                 break # Exit the async for loop
            except Exception as e:
                # Catch any other unexpected errors during message processing loop
                logger.error(f"Unexpected error processing message from {websocket.remote_address}: {e}", exc_info=True) # Log full traceback
                await websocket.send(json.dumps({"detections": [], "error": f"Internal server error: {e}"}))


    # These exceptions handle connection closure outside the message processing loop
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client {websocket.remote_address} disconnected normally (OK).")
    except websockets.exceptions.ConnectionClosedError as e:
         logger.error(f"Client {websocket.remote_address} disconnected with error code {e.code}: {e.reason}")
    except Exception as e:
         # Catch any other unexpected errors in the handler function itself
         logger.error(f"An unexpected error occurred in handler for {websocket.remote_address}: {e}", exc_info=True)
    finally:
        # This block executes when the connection is closed or an unhandled exception occurs
        logger.info(f"Handler finished for {websocket.remote_address}")

# --- Main Server Function ---

async def main():
    """
    Main function to set up and start the WebSocket server.
    """
    # Attempt to get the server's local IP address for informative logging
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) # Connect to a public server (doesn't send data)
        display_ip = s.getsockname()[0]
        s.close()
        logger.info(f"Server likely accessible at IP: {display_ip}")
    except socket.gaierror:
         logger.warning("Could not determine host IP, binding to 0.0.0.0. Use '0.0.0.0' for host.")
         display_ip = "N/A (Binding to 0.0.0.0)"
    except Exception as e:
        logger.warning(f"Could not determine local IP for display: {e}. Binding to 0.0.0.0.")
        display_ip = "N/A (Binding to 0.0.0.0)"


    logger.info(f"Starting WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")

    try:
        # Start the WebSocket server using websockets.serve
        # Pass the correct handler function (detection_handler)
        # Use explicit keyword arguments for host and port
        server = await websockets.serve(
            detection_handler,  # The coroutine function called for each new connection
            host=SERVER_HOST,   # The hostname or IP address to listen on
            port=SERVER_PORT,   # The port number to listen on
            # Optional configuration parameters:
            max_size=20_000_000, # Maximum size of a message in bytes (20MB) - adjust based on image size
            ping_timeout=60,    # Close connection if no pong received within this many seconds
            ping_interval=30    # Send a ping every this many seconds to keep connection alive
            # read_limit=...    # Byte limit for incoming messages buffer
            # write_limit=...   # Byte limit for outgoing messages buffer
        )

        logger.info("Server started successfully.")
        logger.info(f"Listening on ws://{SERVER_HOST}:{SERVER_PORT}") # Use listening host/port in log
        logger.info("Waiting for incoming connections...")

        # Keep the server running indefinitely until interrupted
        await server.wait_closed()

    except OSError as e:
        logger.critical(f"Failed to start server on {SERVER_HOST}:{SERVER_PORT}. Error: {e}")
        logger.critical("Possible reasons: Address already in use, insufficient permissions, or invalid host/port.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred while starting or running the server: {e}", exc_info=True)


# --- Entry Point ---

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down due to KeyboardInterrupt (Ctrl+C).")
    except SystemExit:
         logger.info("Server exiting.") # Handle clean exits from sys.exit()
    except Exception as e:
        logger.critical(f"Unhandled exception in asyncio.run: {e}", exc_info=True)