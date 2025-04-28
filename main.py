# Install required packages
# pip install websockets opencv-python numpy ultralytics

import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import math
import logging
import torch


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

# Load YOLO model
model = YOLO("weights/v9 - 64 epochs.pt")

# Object classes
classNames = ["green", "red", "yellow"]

def distance_finder(focal_length, real_face_width, obj_width_in_frame):
    if obj_width_in_frame == 0:
        return 0
    distance = (real_face_width * focal_length) / obj_width_in_frame
    return distance

async def process_frame(websocket, path):
    logger.info(f"Client connected: {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # Parse the incoming message
                data = json.loads(message)
                frame_data = data.get('image', '')
                
                # Decode base64 image
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    await websocket.send(json.dumps({"detections": []}))
                    continue
                
                # Process with YOLO
                results = model(img, stream=False)
                
                detections = []
                
                # Extract detection results
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values
                        
                        # Calculate object width and distance
                        obj_width = x2 - x1
                        distance = distance_finder(650, 7.1, obj_width)
                        
                        # Confidence
                        confidence = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = classNames[cls]
                        
                        # Add detection to results
                        detections.append({
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "class": class_name,
                            "confidence": confidence,
                            "distance": distance
                        })
                
                # Return detections as JSON
                await websocket.send(json.dumps({"detections": detections}))
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                await websocket.send(json.dumps({"detections": [], "error": str(e)}))
    
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {websocket.remote_address}")

async def main():
    # Start WebSocket server
    server = await websockets.serve(process_frame, "0.0.0.0", 8765)
    logger.info("Server started on ws://0.0.0.0:8765")
    
    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())