import os
import io
import base64
from PIL import Image
import numpy as np
import tensorflow as tf 
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'

# Model paths
SEGMENTATION_MODEL_PATH = 'C:/Users/Atharva Kanchan/Music/Major Project/TB-Detection-DeepLearning/models/unet_model_best.h5'
CLASSIFICATION_MODEL_PATH = 'C:/Users/Atharva Kanchan/Music/Major Project/TB-Detection-DeepLearning/models/densenet_model_best.h5'

# Globals
segmentation_model = None
classification_model = None


# -------------------------
# Load Models
# -------------------------
def load_models():
    global segmentation_model, classification_model
    try:
        segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH)
        classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        segmentation_model = None
        classification_model = None


# -------------------------
# Preprocessing Functions
# -------------------------
def preprocess_for_segmentation(image):
    """Resize to 256x256 for UNet segmentation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image.resize((256, 256))).astype(np.float32) / 255.0

def preprocess_for_classification(image):
    """Resize to 224x224 for DenseNet classification"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image.resize((224, 224))).astype(np.float32) / 255.0


# -------------------------
# Prediction Pipeline
# -------------------------
def predict_tb(image):
    if segmentation_model is None or classification_model is None:
        return {'error': 'Models not loaded properly'}

    try:
        # --- Segmentation ---
        seg_input = preprocess_for_segmentation(image)
        seg_input_exp = np.expand_dims(seg_input, axis=0)   # (1,256,256,3)
        _ = segmentation_model.predict(seg_input_exp)[0]    # mask not displayed, but could be used

        # --- Classification ---
        clf_input = preprocess_for_classification(image)
        clf_input_exp = np.expand_dims(clf_input, axis=0)   # (1,224,224,3)
        tb_probability = classification_model.predict(clf_input_exp)[0][0]

        # Results
        confidence = abs(tb_probability - 0.5) * 2
        result = {
            'tb_probability': float(tb_probability),
            'confidence': float(confidence),
            'prediction': 'TB Detected' if tb_probability > 0.5 else 'Normal',
            'risk_level': 'High' if tb_probability > 0.7 else 'Medium' if tb_probability > 0.3 else 'Low'
        }
        return result

    except Exception as e:
        return {'error': str(e)}


# -------------------------
# Visualization
# -------------------------
def generate_visualization(image, result):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original uploaded X-ray
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title('Original X-ray Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Risk assessment bar chart
    colors = ['green', 'orange', 'red']
    risk_levels = ['Low', 'Medium', 'High']
    risk_values = [0.2, 0.5, 0.8]

    bars = axes[1].bar(risk_levels, risk_values, color=colors, alpha=0.7)
    current_risk = result.get('risk_level', 'Low')
    if current_risk in risk_levels:
        idx = risk_levels.index(current_risk)
        bars[idx].set_alpha(1.0)
        bars[idx].set_edgecolor('black')
        bars[idx].set_linewidth(2)

    axes[1].set_title('Risk Assessment', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Risk Level')
    axes[1].set_ylim(0, 1)

    # Convert to base64 for HTML
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    return plot_data


# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = Image.open(file.stream)

                # Prediction
                result = predict_tb(image)
                if 'error' in result:
                    flash(f'Error processing image: {result["error"]}')
                    return redirect(request.url)

                # Visualization
                plot_data = generate_visualization(image, result)

                # Convert original image to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_data = base64.b64encode(img_buffer.getvalue()).decode()

                return render_template(
                    'results.html',
                    result=result,
                    plot_data=plot_data,
                    img_data=img_data,
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            except Exception as e:
                flash(f'Error processing image: {str(e)}')
                return redirect(request.url)
        else:
            flash('Please upload a valid image file (PNG, JPG, JPEG)')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        image = Image.open(file.stream)
        result = predict_tb(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    load_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
