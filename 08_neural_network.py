import cv2
import pytesseract

if __name__ == '__main__':
    image = cv2.imread("69.jpeg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    result = pytesseract.image_to_string(
        opening,
        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
    )
    print(result)
