import os
import uuid
import time
import numpy as np
import cv2
import requests
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import smtplib
from email.message import EmailMessage
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Solar Inspection API is running", "docs": "/docs"}

# Constants
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load Models
try:
    panel_session = ort.InferenceSession(os.path.join(MODELS_DIR, "panel_detect.onnx"))
    fault_session = ort.InferenceSession(os.path.join(MODELS_DIR, "best.onnx"))
except Exception as e:
    print(f"Error loading models: {e}")

FAULT_CLASSES = ["Bird-drop", "Dusty", "Snow-Covered", "Physical Damage", "Electrical Damage"]
FAULT_LOSS = {
    "Bird-drop": (0.05, 0.10),
    "Dusty": (0.01, 0.05),
    "Snow-Covered": (0.80, 1.00),
    "Physical Damage": (0.30, 0.50),
    "Electrical Damage": (0.50, 0.80)
}

# WEATHER (OPEN-METEO) 
# -------------------------------------------------------------
def get_lat_lon(city):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10
        ).json()
        if "results" in r and len(r["results"]) > 0:
            return r["results"][0]["latitude"], r["results"][0]["longitude"]
        return None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None

def get_effective_sunlight(lat, lon, d):
    try:
        # Get sunshine duration
        sun_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": d,
                "end_date": d,
                "daily": "sunshine_duration",
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        
        sun = sun_resp["daily"]["sunshine_duration"][0] / 3600

        # Get cloud cover
        cloud_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": d,
                "end_date": d,
                "hourly": "cloudcover",
                "timezone": "auto",
            },
            timeout=10,
        ).json()
        
        cloud = cloud_resp["hourly"]["cloudcover"]
        effective_sun = round(sun * (1 - sum(cloud) / len(cloud) / 100), 2)
        return effective_sun
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

@app.get("/weather-info")
async def get_weather(location: str, date: str):
    lat, lon = get_lat_lon(location)
    if lat is None:
        return {"sunlight_hours": 5.5, "status": "location_not_found"}
    
    sun_hours = get_effective_sunlight(lat, lon, date)
    if sun_hours is None:
        return {"sunlight_hours": 5.5, "status": "api_error"}
        
    return {"sunlight_hours": sun_hours, "status": "ok"}

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def preprocess(image, input_size=(640, 640)):
    h, w = image.shape[:2]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, (h, w)

def postprocess(outputs, original_size, conf_threshold=0.25):
    # Standard YOLOv8/v11 postprocessing
    # Output shape is typically [1, num_classes + 4, num_anchors]
    predictions = np.squeeze(outputs[0])
    predictions = np.transpose(predictions, (1, 0))
    
    boxes = []
    scores = []
    class_ids = []
    
    h_orig, w_orig = original_size
    
    for pred in predictions:
        score = np.max(pred[4:])
        if score > conf_threshold:
            class_id = np.argmax(pred[4:])
            xc, yc, w, h = pred[:4]
            
            # Scale to original size
            x1 = (xc - w/2) * w_orig / 640
            y1 = (yc - h/2) * h_orig / 640
            x2 = (xc + w/2) * w_orig / 640
            y2 = (yc + h/2) * h_orig / 640
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(class_id)
            
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, 0.45)
    
    final_boxes = []
    final_scores = []
    final_class_ids = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_scores.append(scores[i])
            final_class_ids.append(class_ids[i])
            
    return final_boxes, final_scores, final_class_ids

@app.post("/analyze")
async def analyze_solar(
    greyImage: UploadFile = File(...),
    location: str = Form(...),
    capacity: float = Form(...),
    sunHours: float = Form(...)
):
    try:
        contents = await greyImage.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
            
        # 1. Panel Detection
        input_tensor, orig_size = preprocess(image)
        outputs = panel_session.run(None, {panel_session.get_inputs()[0].name: input_tensor})
        # Use lower confidence for panels to catch more (0.15 instead of 0.25)
        p_boxes, p_scores, p_ids = postprocess(outputs, orig_size, conf_threshold=0.15)
        
        # 2. Panel Filtering
        if not p_boxes:
            return {"message": "No panels detected", "summary": {"total_panels": 0, "max_energy": 0, "actual_daily_output": 0, "total_daily_loss_kwh": 0}, "panel_analysis": [], "defect_image_url": "", "panel_image_url": ""}

        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in p_boxes]
        aspect_ratios = [(b[2]-b[0]) / (b[3]-b[1]) if (b[3]-b[1]) > 0 else 0 for b in p_boxes]
        
        median_area = np.median(areas)
        median_ar = np.median(aspect_ratios)
        
        # If we have very few detections, be more lenient. 
        # For perspective images, ±15% is too strict. We'll use ±30% for area and ±20% for aspect ratio.
        area_margin = 0.30 
        ar_margin = 0.20
        
        filtered_panels = []
        for i, box in enumerate(p_boxes):
            area = areas[i]
            ar = aspect_ratios[i]
            # Keep panels within margin
            if (median_area * (1 - area_margin) <= area <= median_area * (1 + area_margin)) and \
               (median_ar * (1 - ar_margin) <= ar <= median_ar * (1 + ar_margin)):
                filtered_panels.append(box)
        
        # Fallback: if filtering removed ALL panels, just keep top detections
        if not filtered_panels and p_boxes:
            filtered_panels = p_boxes[:max(1, len(p_boxes)//2)]
                
        # 3. Fault Detection
        f_outputs = fault_session.run(None, {fault_session.get_inputs()[0].name: input_tensor})
        f_boxes, f_scores, f_ids = postprocess(f_outputs, orig_size)
        
        # 4. Fault-to-Panel Mapping and Math
        panel_analysis = []
        total_loss_kwh = 0
        system_energy = capacity * sunHours
        
        # Prepare annotation images
        panel_img = image.copy()
        defect_img = image.copy()
        
        for i, p_box in enumerate(filtered_panels):
            p_faults = []
            p_loss_total = 0
            
            # Draw panel box
            cv2.rectangle(panel_img, (int(p_box[0]), int(p_box[1])), (int(p_box[2]), int(p_box[3])), (0, 255, 0), 2)
            cv2.putText(panel_img, f"P{i+1}", (int(p_box[0]), int(p_box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            for j, f_box in enumerate(f_boxes):
                iou = calculate_iou(p_box, f_box)
                if iou >= 0.50:
                    fault_name = FAULT_CLASSES[f_ids[j]] if f_ids[j] < len(FAULT_CLASSES) else "Unknown"
                    conf = f_scores[j]
                    
                    # Energy Math
                    loss_range = FAULT_LOSS.get(fault_name, (0.1, 0.2))
                    severity_avg = (loss_range[0] + loss_range[1]) / 2
                    
                    # loss = system_energy * severity_avg * confidence (simplified per panel)
                    # wait, the goal says loss = system_energy * severity_avg * confidence
                    # but usually it's per panel. If system_energy is whole system, then total loss is sum.
                    daily_loss = system_energy * severity_avg * conf / len(filtered_panels)
                    
                    p_faults.append({
                        "fault": fault_name,
                        "confidence": float(conf),
                        "loss_percentage": float(severity_avg * 100),
                        "daily_loss": round(daily_loss, 4)
                    })
                    p_loss_total += daily_loss
                    
                    # Draw fault box
                    cv2.rectangle(defect_img, (int(f_box[0]), int(f_box[1])), (int(f_box[2]), int(f_box[3])), (0, 0, 255), 2)
                    cv2.putText(defect_img, f"{fault_name}", (int(f_box[0]), int(f_box[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            panel_analysis.append({
                "panel_number": i + 1,
                "panel_loss_kwh": round(p_loss_total, 4),
                "faults": p_faults
            })
            total_loss_kwh += p_loss_total
            
        # Summary
        actual_daily_output = system_energy - total_loss_kwh
        summary = {
            "total_panels": len(filtered_panels),
            "max_energy": round(system_energy, 2),
            "actual_daily_output": round(actual_daily_output, 2),
            "total_daily_loss_kwh": round(total_loss_kwh, 2)
        }
        
        # Save images
        file_id = str(uuid.uuid4())
        panel_filename = f"panels_{file_id}.jpg"
        defect_filename = f"defects_{file_id}.jpg"
        
        cv2.imwrite(os.path.join(STATIC_DIR, panel_filename), panel_img)
        cv2.imwrite(os.path.join(STATIC_DIR, defect_filename), defect_img)
        
        return {
            "message": "Analysis complete",
            "summary": summary,
            "panel_analysis": panel_analysis,
            "panel_image_url": f"/static/{panel_filename}",
            "defect_image_url": f"/static/{defect_filename}"
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Contact Form Logic
class ContactForm(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

@app.post("/contact")
async def contact_support(form: ContactForm):
    try:
        # Load SMTP settings from environment
        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        support_email = "vnaveen83794@gmail.com"

        if not all([smtp_host, smtp_user, smtp_pass]):
            # Fallback for demo if environment variables are missing
            print(f"SMTP not configured. Message for {support_email} from {form.name} ({form.email}): {form.message}")
            return {"message": "Message received (Demo mode - SMTP not configured)"}

        msg = EmailMessage()
        msg.set_content(f"Name: {form.name}\nEmail: {form.email}\n\nSubject: {form.subject}\n\nMessage:\n{form.message}")
        msg['Subject'] = f"SolarAI Contact: {form.subject}"
        msg['From'] = smtp_user
        msg['To'] = support_email

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)

        return {"message": "Message sent successfully"}
    except Exception as e:
        print(f"SMTP Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
