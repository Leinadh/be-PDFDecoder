import boto3

def processDocument():
    # convert pdf to image is necessary

    # test
    document = 'doc_example.png'

    clientTextract = boto3.client('textract', region_name='us-east-1')
    response = clientTextract.analyze_document(document)
    print('Response textract', response)
    return 'uwu'

    