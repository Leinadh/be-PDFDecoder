!apt-get install poppler-utils 
!pip install pdf2image
!pip install opencv-python
!pip install Pillow

from pdf2image import convert_from_path
import numpy as np
import cv2
import matplotlib.pyplot as plt

#PDF to image 

pdfs = r"Doc35.pdf"
pages = convert_from_path(pdfs, 500)

i = 1
for page in pages:
    image_name = "doc35_page_" + str(i) + ".jpg"  
    page.save(image_name, "JPEG")
    i = i+1   

#Preprocess - CLAHE 

def image_clahe(img_path):
  #Reading the image
  img = cv2.imread(img_path, 1)

  #Converting image to LAB Color model
  lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

  #Splitting the LAB image to different channels
  l, a, b = cv2.split(lab)

  #Applying CLAHE to L-channel
  clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
  cl = clahe.apply(l)

  #Merge the CLAHE enhanced L-channel with the a and b channel
  limg = cv2.merge((cl,a,b))

  #Converting image from LAB Color model to RGB model
  final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
  #cv2_imshow(final)
  
  #Save image 
  output_image_name = "clahe_" + img_path
  cv2.imwrite(output_image_name, final)