import cv2
import numpy as np
import pytesseract
import os
import sys

# Tesseract 경로 설정(사용자마다 다름)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 'crops' 폴더 생성 (이미 존재하면 무시)
if not os.path.exists('crops'):
    os.makedirs('crops')

# 이미지 로드
if len(sys.argv) < 2:
    print("Usage: python image2text.py <image_path>")
    sys.exit(1)

image_file = sys.argv[1]
image_file = os.path.abspath(image_file)  # 절대 경로로 변환

# 이미지 로드
print(f"Attempting to load image from: {image_file}")
large = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
if large is None:
    print("Failed to load image from:", image_file)
    sys.exit(1)

# 이미지 크기 축소
rgb = cv2.pyrDown(large)
small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

# 대비 증가 (contrast adjustment)
contrast = cv2.convertScaleAbs(small, alpha=1.5, beta=0)

# 모폴로지 그라디언트
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
grad = cv2.morphologyEx(contrast, cv2.MORPH_GRADIENT, kernel)

# 이진화 (Otsu thresholding)
_, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 커넥션 클로징 (텍스트 연결 강화)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

# 외곽선 찾기
contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 마스크 생성
mask = np.zeros(bw.shape, dtype=np.uint8)

# 크롭된 영역을 텍스트로 인식하기 위한 처리
index = 0  # 크롭된 이미지 번호
for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    # 텍스트 영역으로 판단될 조건
    if r > 0.45 and w > 8 and h > 8:
        # 크롭된 이미지 저장 (crops 폴더에 저장)
        cropped = rgb[y:y+h, x:x+w]
        cropped_filename = f"crops/cropped_{index}.jpg"
        cv2.imwrite(cropped_filename, cropped)
        
        # 텍스트 인식 (Tesseract 설정 추가)
        text = pytesseract.image_to_string(cropped, lang='eng+kor', config='--psm 6')
        
        # 추출된 텍스트 출력
        if text.strip():
            print(f"Text from cropped_{index}: {text.strip()}")
        else:
            print(f"No text detected in cropped_{index}.")
        
        index += 1

# 결과 이미지 출력 (원하는 경우)
cv2.imshow('rects', rgb)
cv2.waitKey()
cv2.destroyAllWindows()
