import boto3
from textract_utils import process_text_local
from botocore.config import Config
from statistics import mode
import pandas as pd
import matplotlib
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from sklearn.cluster import KMeans
import numpy as np
import glob
import os
from pdf2image import convert_from_path, convert_from_bytes
from get_table import get_table_variables
from get_variables_values import *
import time
import sys


##path = '/home/stevramos/Documentos/HACKATHONBBVA2020/raw_data'
##output_add_path = 'output'

path_json_vars = "variables.json"

split_sep = "<ESC>"

OUTPUT_FORMAT = "png"




def doc_lines_detector(blob_lines, tolerance=0.1):
    '''
    target_image (str) : string object with target image name .png format
    '''
    import sys
    SPACE_WIDTH = 0.1
    prev_top_value = blob_lines[0]['BoundingBox']['Top']
    doc_lines = []
    doc_line_register = []
    n_lines = 0
    current_line = []
    for blob in blob_lines:
        min_box_height = min([line['BoundingBox']['Height']
                              for line in blob_lines[n_lines-1:]])
        if abs(blob['BoundingBox']['Top'] - prev_top_value) >= min_box_height or blob == blob_lines[-1]:
            n_lines += 1
            if blob == blob_lines[-1]:
                if abs(blob['BoundingBox']['Top'] - prev_top_value) >= min_box_height:
                    if len(current_line) > 0:
                        doc_lines.append(split_sep.join(
                            [line[0] for line in current_line]))
                        doc_line_register.append(current_line)
                    doc_lines.append(blob['Text'])
                    doc_line_register.append(
                        [(blob['Text'], blob['BoundingBox'])])
                else:
                    current_line.append((blob['Text'], blob['BoundingBox']))
                    doc_lines.append(split_sep.join(
                        [line[0] for line in current_line]))
                    doc_line_register.append(current_line)
            else:
                doc_lines.append(split_sep.join(
                    [line[0] for line in current_line]))
                doc_line_register.append(current_line)
            current_line = []
            prev_top_value = blob['BoundingBox']['Top']
        current_line.append((blob['Text'], blob['BoundingBox']))
    csv_data = make_csv_with_text_data(doc_lines, doc_line_register)
    csv_with_columns = find_columns(csv_data)
    #dl_ = DOC_FILE.split('.')[0]
    #dl_ = dl_ + "_" + str(index)
    #csv_data.rename(columns={'lines': 'row', 'label': 'column'}).to_csv(
    #    f'image_{dl_}.csv', index=False)
    #plot_image2image(source_image, csv_with_columns, target_image)
    csv_with_columns.rename(
        columns={'lines': 'row', 'label': 'column'}, inplace=True)
    return doc_lines, csv_with_columns


def find_columns(data_blobs_txtract):
    counts = data_blobs_txtract.groupby('lines')['lines'].count()
    first_index = counts[counts == counts.max()].index.min() + 1
    data_blobs_txtract['center_x'] = data_blobs_txtract['left'] + \
        data_blobs_txtract['width']/2
    kmeans = KMeans(n_clusters=counts.max(), random_state=0).fit(
        data_blobs_txtract[['center_x']])
    data_blobs_txtract['label'] = kmeans.labels_
    return data_blobs_txtract


def plot_image2image(image_file, data_blobs_txtract, image_target_file):
    cols_colors = list(matplotlib.colors.cnames.values())
    source_img = Image.open(image_file).convert("RGBA")
    FULL_WIDTH, FULL_HEIGHT = source_img.size
    draw = ImageDraw.Draw(source_img)
    for label_index in range(data_blobs_txtract.label.max() + 1):
        for _, row in data_blobs_txtract[data_blobs_txtract.label == label_index].iterrows():
            x0 = (row['left'])*FULL_WIDTH
            y0 = (row['top'])*FULL_HEIGHT
            x1 = (row['left'] + row['width'])*FULL_WIDTH
            y1 = (row['top'] + row['height'])*FULL_HEIGHT
            draw.rectangle(((x0, y0), (x1, y1)), fill=cols_colors[label_index])

    #print (image_target_file)
    Image.fromarray(np.hstack((np.array(Image.open(image_file).convert(
        "RGBA")), np.array(source_img)))).save(image_target_file, OUTPUT_FORMAT)
    return None


def make_csv_with_text_data(doc_lines, doc_line_register):
    left = []
    top = []
    height = []
    width = []
    texts = []
    lines = []
    for i, line in enumerate(doc_lines):
        for j, text in enumerate(line.split(split_sep)):
            texts.append(text)
            boundingbox = doc_line_register[i][j][1]
            left.append(boundingbox['Left'])
            top.append(boundingbox['Top'])
            height.append(boundingbox['Height'])
            width.append(boundingbox['Width'])
            lines.append(i)
    data_image = pd.DataFrame(
        {'texts': texts, 'left': left, 'top': top, 'height': height, 'width': width, 'lines': lines})
    return data_image



def analyze_text_pdf(DOC_FILE, document):
    #files_in_dir = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #for filename in files_in_dir:
    #    start_time = time.time()
    #    print (f"DOC_FILE: {filename}"),
    #    DOC_FILE = filename
    #    document = os.path.join(path, DOC_FILE)

    #images = convert_from_bytes(open(document, 'rb').read())

    if DOC_FILE.split(".")[-1] == "pdf":
        images = convert_from_bytes(document)
    else:
        #pasar a imagen
        images = []
        images.append(document)

    dict_variable_doc = {}

    for i, image in enumerate(images):
        image = images[i]
        document_name = DOC_FILE.split('.')[0]
        
        #source_file = os.path.join(path, output_add_path, f"{document_name}_0{i}.{OUTPUT_FORMAT}")
        #target_file = os.path.join(path,output_add_path,f'{document_name}__0{i}_comparison.{OUTPUT_FORMAT}')
        #----#  
        #image.save(source_file)
        _, lines = process_text_local(image, OUTPUT_FORMAT)
        #print ('miau')
        lines, csv_with_columns = doc_lines_detector(lines)
        #print (target_file)
        #----------------- Data extraction ----------------------#
        df_doc_data = get_table_variables(csv_with_columns.sort_values('row').reset_index(drop=True))
        
        with open(path_json_vars, 'r') as j:
            dict_parameters = json.load(j)
        
        df_doc_data = processing_text(df_doc_data)
        dict_variables = get_variables_index(df_doc_data, dict_parameters, sequence_matcher_similarity)
        dict_variables = get_dict_vars_values(df_doc_data, dict_variables)
        dict_variable_doc.update(dict_variables)
        #------------------ Print lines --------------------------#

    dict_variable_doc = processing_values_dict(dict_variable_doc)
    dict_variable_doc["DOCUMENTO"] = document_name

    # with open(os.path.join(path,output_add_path,f'{document_name}_0{i}_final.json'), 'w') as file:
    #    json.dump(dict_variable_doc, file)

    return dict_variable_doc
    
    

def processDocument(image_blob=None):

    # test
    if image_blob is None:
        document = 'doc_example.png'
        with open(document, 'rb') as f:
            image_blob = f.read()

    # convert pdf to image

    clientTextract = boto3.client('textract', region_name='us-east-1', aws_access_key_id='AKIAI245TDM5XGD2TWNA',
                                  aws_secret_access_key='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')
    response = clientTextract.detect_document_text(
        Document={'Bytes': image_blob})

    # process response

    return response