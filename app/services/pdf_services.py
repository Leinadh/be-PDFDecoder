import boto3
# import os

def processDocument(image_blob=None):

    # test
    if image_blob is None:
        document = 'doc_example.png'
        with open(document, 'rb') as f:
            image_blob = f.read()

    # convert pdf to image

    clientTextract = boto3.client('textract', region_name='us-east-1', aws_access_key_id='AKIAI245TDM5XGD2TWNA', aws_secret_access_key='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')
    response = clientTextract.detect_document_text(Document={'Bytes':image_blob})

    # process response 

    return response

    