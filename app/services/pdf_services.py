import boto3
import os

def processDocument():
    # convert pdf to image is necessary

    # test
    # print('wd:: ', os.getcwd())
    document = 'doc_example.png'
    with open(document, 'rb') as f:
        image_blob = f.read()

    clientTextract = boto3.client('textract', region_name='us-east-1')
    response = clientTextract.detect_document_text(Document={'Bytes':image_blob})
    print('Response textract', response)
    return 'uwu'

    