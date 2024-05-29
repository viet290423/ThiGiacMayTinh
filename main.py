import os
from flask import Flask, request, render_template, redirect, url_for
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Tải mô hình đã huấn luyện
model = load_model('model5.h5')

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []
    
    # GaussianBlur -> Giảm nhiễu
    img_gau = cv2.GaussianBlur(image, (5,5), 0)
    # Thresholding -> binary
    _, thresh = cv2.threshold(img_gau, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Tạo kernel
    kernel = np.ones((5,5),np.uint8)
    # mophorlogicalX -> loại bỏ điểm nhiễu nhỏ nếu cần 
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # tìm contour -> thiết kế thuật toán dể nhận diện contours là số (dựa vào area hoặc arclength)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    


    rois = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > 10 and w > 10:
            roi = thresh[y:y+h, x:x+w]
            rois.append((x, roi))

        rois = sorted(rois, key=lambda d: d[0])

    processed_digits = []
    for (x, roi) in rois:
        height, width = roi.shape
        if height > width:
            padding = (height - width) // 2
            roi = cv2.copyMakeBorder(roi, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
        else:
            padding = (width - height) // 2
            roi = cv2.copyMakeBorder(roi, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = roi.astype('float32') / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        processed_digits.append(roi)

    predictions = []
    for digit in processed_digits:
        prediction = model.predict(digit)
        predicted_digit = np.argmax(prediction)
        predictions.append(str(predicted_digit))

    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            predictions = process_image(file_path)
            phone_number = ''.join(predictions)
            return render_template('index.html', phone_number=phone_number, image_path=file.filename)
    return render_template('index.html', phone_number='', image_path='')

if __name__ == '__main__':
    app.run(debug=True)
