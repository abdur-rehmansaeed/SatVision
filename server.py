import os
import json
import numpy as np
import tifffile as tiff
import cv2
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import ee
import requests
from datetime import datetime, timedelta
from flask import Flask, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import math
from google import genai
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)
CORS(app)

@app.route("/")
def health_check():
    return "✅ SatVision Backend is alive!", 200

# --- 1. CONFIGURATION ---
CHECKPOINT_PATH = "./models/WFV2_unetplusplus-epoch=09-val_bce_land_water=0.3539.ckpt"
DOWNLOADS_DIR = "server_downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. GOOGLE EARTH ENGINE SETUP ---
print("🌍 Initializing Earth Engine...", flush=True)
try:
    credentials_json = os.environ.get("EE_CREDENTIALS")
    if credentials_json:
        key_dict = json.loads(credentials_json)
        from google.oauth2 import service_account
        credentials = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=['https://www.googleapis.com/auth/earthengine']
        )
        ee.Initialize(credentials, project='gen-lang-client-0114614261')
        print("✅ Earth Engine Linked via Service Account.", flush=True)
    else:
        ee.Initialize(project='gen-lang-client-0114614261')
        print("✅ Earth Engine Linked via Local Auth.", flush=True)
except Exception as e:
    print(f"❌ Earth Engine Init Failed: {e}", flush=True)

# --- 3. MODEL DEFINITION ---
class DynamicEvalModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_params = kwargs.get('model_params', {'model_type': 'unetplusplus', 'encoder_name': 'resnet34', 'in_channels': 15, 'out_classes': 2})
        self._build_model()

    def _build_model(self):
        self.model = smp.UnetPlusPlus(
            encoder_name=self.model_params.get('encoder_name', 'resnet34'),
            encoder_weights=None, 
            in_channels=self.model_params.get('in_channels', 15),
            classes=self.model_params.get('out_classes', 2),
            activation=None
        )

    def forward(self, x):
        return self.model(x)

print("⏳ Loading model...", flush=True)
try:
    model = DynamicEvalModel.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ Model Loaded Successfully!", flush=True)
except Exception as e:
    print(f"❌ Model Load Failed: {e}", flush=True)
    model = None

# --- 4. UTILS ---
def predict_water_mask(tiff_path):
    if not model or not os.path.exists(tiff_path): return None

    image = tiff.imread(tiff_path).astype(np.float32)
    if image.ndim == 3 and image.shape[0] == 15:
        image = np.transpose(image, (1, 2, 0))

    image_tensor = np.transpose(image, (2, 0, 1))
    image_tensor = np.clip(image_tensor / 10000.0, 0, 1) if np.max(image_tensor) > 1.0 else np.clip(image_tensor, 0, 1)

    tile_size, overlap = 256, 0.5
    stride = int(tile_size * (1 - overlap))
    c, h, w = image_tensor.shape
    
    pad_h = (tile_size - (h % stride)) % stride + (tile_size - stride) if h % stride != 0 else 0
    pad_w = (tile_size - (w % stride)) % stride + (tile_size - stride) if w % stride != 0 else 0
    image_padded = np.pad(image_tensor, ((0,0), (0, pad_h), (0, pad_w)), mode='reflect')
    
    padded_h, padded_w = image_padded.shape[1], image_padded.shape[2]
    prob_map = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.float32)

    y_steps = list(range(0, padded_h - tile_size + 1, stride))
    x_steps = list(range(0, padded_w - tile_size + 1, stride))

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                tile = torch.from_numpy(image_padded[:, y:y+tile_size, x:x+tile_size]).unsqueeze(0).to(DEVICE)
                logits = model(tile)
                probs = torch.sigmoid(logits)
                water_prob = probs[0, 1, :, :].cpu().numpy() if probs.shape[1] > 1 else probs[0, 0, :, :].cpu().numpy()
                prob_map[y:y+tile_size, x:x+tile_size] += water_prob
                count_map[y:y+tile_size, x:x+tile_size] += 1.0

    final_prob = (prob_map / count_map)[:h, :w]
    
    # --- EDGE SNAPPING (Guided Filter) ---
    ref_img = image / 10000.0 if np.max(image) > 1.0 else image.copy()
    rgb = np.clip(ref_img[:, :, [1, 2, 3]] * 3.0, 0, 1) * 255
    
    rgb_32f = rgb.astype(np.float32) / 255.0
    prob_32f = final_prob.astype(np.float32)
    
    try:
        refined_prob = cv2.ximgproc.guidedFilter(guide=rgb_32f, src=prob_32f, radius=10, eps=0.01)
        clean_mask = (refined_prob > 0.5).astype(np.uint8)
    except Exception as e:
        print(f"⚠️ Guided Filter Failed: {e}. Falling back to standard thresholding.", flush=True)
        clean_mask = (final_prob > 0.5).astype(np.uint8)
        clean_mask = cv2.medianBlur(clean_mask, 5)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 500
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            cv2.drawContours(clean_mask, [cnt], -1, 0, -1)

    return clean_mask * 255

def create_rgb_preview(tiff_path, filename):
    if not os.path.exists(tiff_path): return None
    
    image = tiff.imread(tiff_path).astype(np.float32)

    if image.ndim == 2:
        rgb = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        rgb[image == 255] = [255, 0, 0] 
        save_path = os.path.join(DOWNLOADS_DIR, filename)
        cv2.imwrite(save_path, rgb)
        return filename

    if image.shape[0] == 15: image = np.transpose(image, (1, 2, 0))
    if np.max(image) > 1.0: image = image / 10000.0
    
    rgb = image[:, :, [1, 2, 3]]
    rgb = np.clip(rgb * 3.0, 0, 1) * 255
    rgb = rgb.astype(np.uint8)
    
    gaussian = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    sharpened = cv2.addWeighted(rgb, 1.5, gaussian, -0.5, 0)
    
    save_path = os.path.join(DOWNLOADS_DIR, filename)
    cv2.imwrite(save_path, sharpened)
    return filename

def create_overlay_png(mask, color, filename):
    if mask is None: return None
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    b, g, r = color
    rgba[mask == 255, 0] = b
    rgba[mask == 255, 1] = g
    rgba[mask == 255, 2] = r
    rgba[mask == 255, 3] = 180
    save_path = os.path.join(DOWNLOADS_DIR, filename)
    cv2.imwrite(save_path, rgba)
    return filename

def download_gee_tile(image, n, s, e, w, prefix, r, c):
    region = ee.Geometry.BBox(w, s, e, n)
    url = image.getDownloadURL({
        'crs': 'EPSG:4326',
        'region': region,
        'scale': 10,
        'format': 'GEO_TIFF'
    })
    resp = requests.get(url, timeout=120)
    if resp.status_code != 200: 
        print(f"❌ Tile {r}-{c} failed.", flush=True)
        return None
        
    path = os.path.join(DOWNLOADS_DIR, f"{prefix}_tile_{r}_{c}.tiff")
    with open(path, 'wb') as f: 
        f.write(resp.content)
    return path

def fetch_satellite_image(coords, time_interval, filename_prefix):
    try:
        region = ee.Geometry.BBox(coords['west'], coords['south'], coords['east'], coords['north'])
        start_date, end_date = time_interval

        col_l2a = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(region)
                   .filterDate(start_date, end_date)
                   .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 75)))

        if col_l2a.size().getInfo() > 0:
            print(f"   🛰️ Using Sentinel-2 L2A for {filename_prefix}...", flush=True)
            first_img = col_l2a.sort('CLOUDY_PIXEL_PERCENTAGE', False).first()
            actual_date = ee.Date(first_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            source_type = "Sentinel-2 L2A (Surface Reflectance)"
            image = col_l2a.sort('CLOUDY_PIXEL_PERCENTAGE', False).mosaic() 

            bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12']
            base = image.select(bands).toInt16()

            cloud_prob = image.expression('b("B2") / (b("B2") + b("B11") + 0.0001)') \
                              .multiply(10000).toInt16().rename('cloud_prob')
            water_map = image.normalizedDifference(['B3','B8']).gt(0).unmask(0) \
                             .multiply(10000).toInt16().rename('water_map')

            p1 = base.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9'])
            b10_zero = base.select('B1').multiply(0).toInt16().rename('B10')
            p2 = base.select(['B11','B12'])
            final_image = ee.Image.cat([p1, b10_zero, p2, cloud_prob, water_map])

        else:
            print(f"   ⚠️ L2A unavailable → L1C for {filename_prefix}...", flush=True)
            col_l1c = (ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
                       .filterBounds(region)
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 75)))

            if col_l1c.size().getInfo() == 0:
                return None, None, None

            first_img = col_l1c.sort('CLOUDY_PIXEL_PERCENTAGE', False).first()
            actual_date = ee.Date(first_img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            source_type = "Sentinel-2 L1C (Top of Atmosphere)"
            image = col_l1c.sort('CLOUDY_PIXEL_PERCENTAGE', False).mosaic()

            bands = ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
            base = image.select(bands).toInt16()

            cloud_prob = image.expression('(b("B2")+b("B10")) / (b("B2")+b("B10")+b("B11")+0.0001)') \
                              .multiply(10000).toInt16().rename('cloud_prob')
            water_map = image.normalizedDifference(['B3','B8']).gt(0).unmask(0) \
                             .multiply(10000).toInt16().rename('water_map')

            final_image = ee.Image.cat([base, cloud_prob, water_map])

        final_image = final_image.unmask(0)

        MAX_DEG_STEP = 0.08
        n, s, e, w = coords['north'], coords['south'], coords['east'], coords['west']

        rows = math.ceil((n - s) / MAX_DEG_STEP)
        cols = math.ceil((e - w) / MAX_DEG_STEP)

        lat_steps = np.linspace(n, s, rows + 1)
        lon_steps = np.linspace(w, e, cols + 1)

        tile_arrays = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                tile_n, tile_s = lat_steps[r], lat_steps[r+1]
                tile_w, tile_e = lon_steps[c], lon_steps[c+1]
                
                tile_path = download_gee_tile(final_image, tile_n, tile_s, tile_e, tile_w, filename_prefix, r, c)
                if not tile_path: return None, None, None
                
                img = tiff.imread(tile_path)
                row_list.append(img)
                os.remove(tile_path)
                
            min_h = min(tile.shape[0] for tile in row_list)
            row_list = [tile[:min_h, :, :] for tile in row_list]
            row_stitched = np.concatenate(row_list, axis=1)
            tile_arrays.append(row_stitched)

        min_w = min(row.shape[1] for row in tile_arrays)
        tile_arrays = [row[:, :min_w, :] for row in tile_arrays]
        final_stitched = np.concatenate(tile_arrays, axis=0)

        save_path = os.path.join(DOWNLOADS_DIR, f"{filename_prefix}.tiff")
        tiff.imwrite(save_path, final_stitched)
        return save_path, actual_date, source_type

    except Exception as e:
        print(f"❌ GEE Error {filename_prefix}: {e}", flush=True)
        return None, None, None

def fetch_dynamic_world_baseline(coords, time_interval, filename_prefix):
    try:
        region = ee.Geometry.BBox(coords['west'], coords['south'], coords['east'], coords['north'])
        start_date, end_date = time_interval

        dw_col = (ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
                  .filterBounds(region)
                  .filterDate(start_date, end_date))
        
        if dw_col.size().getInfo() == 0:
            print(f"❌ No Dynamic World data found for {filename_prefix}", flush=True)
            return None, None

        final_image = dw_col.select('label').mode().eq(0).unmask(0).multiply(255).toByte()

        MAX_DEG_STEP = 0.08
        n, s, e, w = coords['north'], coords['south'], coords['east'], coords['west']

        rows = math.ceil((n - s) / MAX_DEG_STEP)
        cols = math.ceil((e - w) / MAX_DEG_STEP)

        lat_steps = np.linspace(n, s, rows + 1)
        lon_steps = np.linspace(w, e, cols + 1)

        tile_arrays = []
        for r in range(rows):
            row_list = []
            for c in range(cols):
                tile_n, tile_s = lat_steps[r], lat_steps[r+1]
                tile_w, tile_e = lon_steps[c], lon_steps[c+1]
                
                tile_path = download_gee_tile(final_image, tile_n, tile_s, tile_e, tile_w, filename_prefix, r, c)
                if not tile_path: return None, None
                
                img = tiff.imread(tile_path)
                row_list.append(img)
                os.remove(tile_path)
                
            min_h = min(tile.shape[0] for tile in row_list)
            row_list = [tile[:min_h, ...] for tile in row_list]
            row_stitched = np.concatenate(row_list, axis=1)
            tile_arrays.append(row_stitched)

        min_w = min(row.shape[1] for row in tile_arrays)
        tile_arrays = [row[:, :min_w, ...] for row in tile_arrays]
        final_stitched = np.concatenate(tile_arrays, axis=0)

        save_path = os.path.join(DOWNLOADS_DIR, f"{filename_prefix}.tiff")
        tiff.imwrite(save_path, final_stitched)
        return save_path, "Google Dynamic World (30-Day Median)"

    except Exception as e:
        print(f"❌ Dynamic World Error {filename_prefix}: {e}", flush=True)
        return None, None
    
# --- 5. GEMINI MULTIMODAL REPORT GENERATION ---  
def generate_flood_report(stats, coords, dates, image_path):      
    try:          
        gemini_api_key = os.environ.get("GEMINI_API_KEY")           
        if not gemini_api_key:              
            return "⚠️ AI Report unavailable: GEMINI_API_KEY missing. Please add it to your space secrets."  

        # Initialize the modern Gemini Client
        client = genai.Client(api_key=gemini_api_key)
        
        # Load the image we just created so Gemini can see it
        img = Image.open(image_path)
        
        prompt = f"""          
        You are an expert hydrologist and disaster response analyst.           
        I have provided an image overlay of the flooded region (red areas indicate new flooding) along with calculated data.  

        Location Bounding Box: {coords}          
        Analysis Period: {dates['baseline']} to {dates['latest']}  

        Key Metrics:          
        - Pre-flood Water Area: {stats['baseline_sq_km']:.2f} sq km          
        - Current Total Water Area: {stats['current_sq_km']:.2f} sq km          
        - New Flooded Area: {stats['flood_sq_km']:.2f} sq km  

        Look closely at the provided flood map image and the metrics above. 
        Write a concise, professional flood assessment report.  

        Structure the report with the following headers:          
        1. Executive Summary          
        2. Visual & Impact Assessment (Discuss the distribution of the red flooded areas you see in the image)          
        3. Recommendations for Responders  

        Keep it factual, professional, and under 300 words. Do not make up external data.          
        """  

        # Pass BOTH the image and prompt using the updated v2.5 Flash model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[img, prompt]
        )          
        return response.text      
    except Exception as e:          
        print(f"❌ Gemini Error: {e}", flush=True)          
        return "⚠️ AI Report generation failed due to server error."

@app.route('/api/detect_stream', methods=['POST'])
def detect_stream():
    data = request.json
    coords = data.get('bbox')
    
    def generate_process():
        yield json.dumps({"progress": 5, "log": "🌍 Locking Coordinates..."}) + "\n"
        
        today = datetime.now()
        
        interval_latest = ((today - timedelta(days=10)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'))
        
        baseline_end = today - timedelta(days=10)
        interval_baseline = ((baseline_end - timedelta(days=30)).strftime('%Y-%m-%d'), baseline_end.strftime('%Y-%m-%d'))
        
        yield json.dumps({"progress": 20, "log": "📡 Fetching Post-Flood Data..."}) + "\n"
        path_latest, date_latest, src_latest = fetch_satellite_image(coords, interval_latest, "latest")
        if not path_latest:
            yield json.dumps({"error": "Failed to fetch latest image (Cloudy/No Data)"}) + "\n"
            return
            
        yield json.dumps({"progress": 40, "log": "📡 Fetching Dynamic World Baseline..."}) + "\n"
        path_pre, src_pre = fetch_dynamic_world_baseline(coords, interval_baseline, "previous")
        date_pre = f"{interval_baseline[0]} to {interval_baseline[1]}"

        yield json.dumps({"progress": 60, "log": "🧠 Analyzing Current Water..."}) + "\n"
        mask_latest = predict_water_mask(path_latest)

        yield json.dumps({"progress": 80, "log": "🧠 Loading Baseline Water..."}) + "\n"
        if path_pre:
            mask_pre = tiff.imread(path_pre).astype(np.uint8)
        else:
            mask_pre = np.zeros_like(mask_latest)

        yield json.dumps({"progress": 90, "log": "🌊 Generating Flood Map..."}) + "\n"
        
        if mask_latest.shape != mask_pre.shape:
             mask_pre = cv2.resize(mask_pre, (mask_latest.shape[1], mask_latest.shape[0]), interpolation=cv2.INTER_NEAREST)

        flood_mask = cv2.bitwise_and(mask_latest, cv2.bitwise_not(mask_pre))  
        
        base_url = request.host_url.rstrip('/')  
        
        # 1. CREATE IMAGES FIRST (So Gemini can see them)
        yield json.dumps({"progress": 91, "log": "🎨 Generating Visual Overlays..."}) + "\n"
        url_latest = create_overlay_png(mask_latest, [255, 0, 0], "overlay_latest.png")          
        url_pre = create_overlay_png(mask_pre, [255, 255, 0], "overlay_previous.png")          
        url_flood = create_overlay_png(flood_mask, [0, 0, 255], "overlay_flood.png")  
        rgb_latest = create_rgb_preview(path_latest, "rgb_latest.jpg")          
        rgb_pre = create_rgb_preview(path_pre, "rgb_previous.jpg")  

        # 2. CALCULATE METRICS        
        yield json.dumps({"progress": 92, "log": "📊 Calculating Spatial Metrics..."}) + "\n"  
        pixel_to_sq_km = 100 / 1_000_000          
        stats = {              
            "baseline_sq_km": np.count_nonzero(mask_pre == 255) * pixel_to_sq_km,              
            "current_sq_km": np.count_nonzero(mask_latest == 255) * pixel_to_sq_km,              
            "flood_sq_km": np.count_nonzero(flood_mask == 255) * pixel_to_sq_km,          
        }          
        dates = {"baseline": date_pre, "latest": date_latest}  

        # 3. GENERATE GEMINI REPORT WITH IMAGE
        yield json.dumps({"progress": 95, "log": "🤖 Generating AI Assessment Report..."}) + "\n"  
        
        # Get the path to the red flood map we just created
        flood_image_path = os.path.join(DOWNLOADS_DIR, "overlay_flood.png")
        
        # Pass the image path to our new Gemini function
        ai_report_text = generate_flood_report(stats, coords, dates, flood_image_path)          
        
# 4. SAVE THE REPORT (AS A FORMATTED PDF)
        yield json.dumps({"progress": 98, "log": "📄 Compiling PDF Report..."}) + "\n"
        
        unique_id = int(datetime.now().timestamp())
        report_filename = f"flood_assessment_report_{unique_id}.pdf"          
        report_path = os.path.join(DOWNLOADS_DIR, report_filename)  

        # Create a beautiful composite image for the PDF (Actual Satellite + Red Flood overlay)
        pdf_img_path = os.path.join(DOWNLOADS_DIR, f"pdf_img_{unique_id}.jpg")
        base_img = cv2.imread(os.path.join(DOWNLOADS_DIR, "rgb_latest.jpg"))
        
        if base_img is not None:
            pdf_img = base_img.copy()
            pdf_img[flood_mask == 255] = (0, 0, 255) # Add bright red pixels for the flood
            cv2.imwrite(pdf_img_path, pdf_img)
        else:
            pdf_img_path = flood_image_path # Fallback

        # Draw the PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add Title
        pdf.set_font("helvetica", "B", 18)
        pdf.cell(0, 15, "SatVision Flood Assessment Report", align="C", ln=True)
        pdf.ln(5)
        
        # Add the Image
        pdf.image(pdf_img_path, x=15, w=180)
        pdf.ln(10)
        
        # Add the Text (fpdf2's markdown=True automatically bolds Gemini's headers!)
        pdf.set_font("helvetica", size=11)
        pdf.multi_cell(0, 6, text=ai_report_text, markdown=True)
        
        pdf.output(report_path)
        
        yield json.dumps({
            "progress": 100,
            "log": "✅ Detection Complete!",
            "result": {
                "latest": f"{base_url}/mask/{url_latest}",
                "previous": f"{base_url}/mask/{url_pre}",
                "flood": f"{base_url}/mask/{url_flood}"
            },
            "meta": {
                "latest_date": date_latest,
                "previous_date": date_pre,
                "latest_source": src_latest,
                "previous_source": src_pre,
                "latest_rgb": f"{base_url}/mask/{rgb_latest}",
                "previous_rgb": f"{base_url}/mask/{rgb_pre}"
            },
            "report": {
                "text": ai_report_text,
                "download_url": f"{base_url}/mask/{report_filename}",
                "metrics": stats
            }
        }) + "\n"
        
    return Response(stream_with_context(generate_process()), mimetype='application/json')

@app.route('/mask/<filename>')
def serve_mask(filename):
    return send_from_directory(DOWNLOADS_DIR, filename)

if __name__ == '__main__':
    print("🚀 Flood Backend Running...", flush=True)
    app.run(host="0.0.0.0", port=7860, threaded=True)