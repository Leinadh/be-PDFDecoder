import boto3
# import os

def processDocument(image_blob=None):
    # convert pdf to image

    # test
    document = 'doc_example.png'
    with open(document, 'rb') as f:
        image_blob = f.read()

    clientTextract = boto3.client('textract', region_name='us-east-1')
    response = clientTextract.detect_document_text(Document={'Bytes':image_blob})
    return response

    