import boto3
import io
from io import BytesIO
import sys
from credentials import aws_access_key_id, aws_secret_access_key
from config import region
import math
from PIL import Image, ImageDraw, ImageFont

def ShowBoundingBox(draw,box,width,height,boxColor):
             
    left = width * box['Left']
    top = height * box['Top'] 
    draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline=boxColor)

def TexTractAnalysis(client,image,image_binary, show_image = True):
    lines = []
    response = client.detect_document_text(Document={'Bytes':image_binary})
    ##response = client.analyze_document(Document={'Bytes': image_binary},
      ##  FeatureTypes=["TABLES", "FORMS"])
    #Get the text blocks
    blocks=response['Blocks']
    width, height =image.size  
    draw = ImageDraw.Draw(image)
    for block in blocks:
        draw=ImageDraw.Draw(image)
        if block['BlockType'] == 'LINE':
            ShowBoundingBox(draw, block['Geometry']['BoundingBox'],width,height, 'blue')
            lines.append({'Text':block['Text'], 'BoundingBox': block['Geometry']['BoundingBox']})
    if show_image:
        # Display the image
        image.show()
    return len(blocks), lines
def process_text_analysis_cloud(bucket, document):
    #Get the document from S3
    s3_connection = boto3.resource('s3', aws_access_key_id = aws_access_key_id, aws_secret_access_key= aws_secret_access_key)    
    s3_object = s3_connection.Object(bucket,document)
    s3_response = s3_object.get()
    stream = io.BytesIO(s3_response['Body'].read())
    image=Image.open(stream)
    # Analyze the document
    client = boto3.client('textract', aws_access_key_id = aws_access_key_id, aws_secret_access_key= aws_secret_access_key, region_name = region)
    image_binary = stream.getvalue()
    return TexTractAnalysis(client, image, image_binary, show_image = True)

def process_text_local(PIL_IMAGE, img_format, show_image = False):
    # Convert image to bytes
    stream = io.BytesIO()
    PIL_IMAGE.save(stream, img_format)
    image_binary = stream.getvalue()
    client = boto3.client('textract', aws_access_key_id = aws_access_key_id, aws_secret_access_key= aws_secret_access_key, region_name = region)
    return TexTractAnalysis(client,PIL_IMAGE,image_binary, show_image = show_image)