import cv2
import numpy as np
from pyzbar.pyzbar import decode

def decoder(image):
	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	barcode = decode(gray_img)

	for obj in barcode:
		points = obj.polygon
		(x, y, w, h) = obj.rect
		pts = np.array(points, np.int32)
		pts = pts.reshape((-1, 1, 2))
		cv2.polylines(image, [pts], True, (0, 255, 0), 3)

		barcodeData = obj.data.decode("utf-8")
		barcodeType = obj.type
		string = "Data " + str(barcodeData) + " | Type " + str(barcodeType)

		cv2.putText(image, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
		print("Barcode: " + barcodeData + " | Type: " + barcodeType)


img = cv2.imread("zbar-test.jpg")
decoder(img)
cv2.imshow("Barcode_image", img)
cv2.waitKey(0)





