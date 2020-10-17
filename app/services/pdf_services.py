import boto3

def processDocument():
    # convert pdf to image is necessary

    # test
    document = 'doc_example.png'
    with open('document') as f:
        image_blob = f.read()

    clientTextract = boto3.client('textract', region_name='us-east-1')
    response = clientTextract.detect_document_text(Document={'Bytes':image_blob})
    print('Response textract', response)
    return 'uwu'

    