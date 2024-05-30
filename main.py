import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the pre-trained model
model = load_model('model5.h5')

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    
    # Gaussian Blur -> giảm nhiễu 
    img_gau = cv2.GaussianBlur(image, (5, 5), 0)
    # Thresholding -> binary
    _, thresh = cv2.threshold(img_gau, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Tạo kernel
    kernel = np.ones((5, 5), np.uint8)
    # mophorlogicalX -> loại bỏ điểm nhiễu nhỏ
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Tìm contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 10:
            roi = thresh[y:y + h, x:x + w]
            rois.append((x, roi))

    # Sort contours from left to right
    rois = sorted(rois, key=lambda d: d[0])

    predictions = []
    for _, roi in rois:
        h, w = roi.shape
        # Tính toán padding để tạo ra hình vuông rois
        pad_height = (w - h) // 2 if w > h else 0
        pad_width = (h - w) // 2 if h > w else 0
        
        # Sử dụng padding
        roi = cv2.copyMakeBorder(roi, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=0)
        
        # Resize về 28x28
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        
        # Predict the digit
        pred = model.predict(roi)
        predicted_labels = np.argmax(pred)
        predictions.append(str(predicted_labels))

    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        predictions = process_image(file_path)
        phone_number = ''.join(predictions)
        return render_template('index.html', phone_number=phone_number, image_path=file.filename)
    
    return render_template('index.html', phone_number='', image_path='')

if __name__ == '__main__':
    app.run(debug=True)