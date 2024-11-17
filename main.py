from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import time

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_path: str = Form(...)):
    
    try:
        # Muat model TensorFlow Lite dari model_path yang diberikan
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Mendapatkan informasi input dan output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_shape = input_details[0]['shape']  # [1, 224, 224, 1] untuk gambar grayscale

        # Periksa format file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File harus berupa gambar")
        
        # Baca file gambar dan konversi ke grayscale
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Konversi ke grayscale
        
        # Ubah ukuran gambar ke [224, 224]
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized).astype(np.float32)
        
        # Normalisasi dan tambahkan dimensi batch dan channel
        image_normalized = image_array / 255.0
        image_normalized = np.expand_dims(image_normalized, axis=-1)  # Channel untuk grayscale
        image_normalized = np.expand_dims(image_normalized, axis=0)   # Batch size

        # Jalankan model untuk inferensi
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], image_normalized)
        interpreter.invoke()
        end_time = time.time()

        # Dapatkan hasil prediksi
        output = interpreter.get_tensor(output_details[0]['index'])
        confidence = float(output[0][0])  # Probabilitas untuk "Ada TB"

        # Tentukan hasil prediksi
        prediction = "Ada TB" if confidence > 0.5 else "Tidak Ada TB"
        
        # Kembalikan hasil prediksi dalam JSON
        return JSONResponse(content={
            "prediction": prediction,
            "confidence": confidence,
            "inference_time": round(end_time - start_time, 4)  # Dalam detik
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam memproses gambar: {str(e)}")
