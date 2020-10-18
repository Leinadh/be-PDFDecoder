#!pip install fuzzywuzzy
#!pip install unidecode

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import re
import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unidecode
from fuzzywuzzy import fuzz
import pandas as pd
import json
from copy import deepcopy
import difflib


def no_space_start_end(x):
  return re.sub("^[\\s]*|[\\s]*$",'', str(x)) if pd.notnull(x) else ""

def no_space_between(x):
  return re.sub("[\\s]+",' ', str(x)) if pd.notnull(x) else ""

def upper_case(x):
  return str(x).upper()

def lower_case(x):
  return str(x).lower()

def no_alphanum(x):
  return re.sub("[^\w|^\s]", " ", str(x)) if pd.notnull(x) else ""

def no_numeric(x):
  return re.sub("\d", " ", str(x)) if pd.notnull(x) else ""

def no_stopwords(x):
  if pd.notnull(x):
    stop_words=set(stopwords.words('spanish'))
    word_tokens = word_tokenize(str(x))
    return ' '.join([w for w in word_tokens if not w in stop_words]) 
  else:
    return ""

def words_unidecode(x):
  return unidecode.unidecode(str(x)) if pd.notnull(x) else ""
  
  
def processing_text(df):
  _df = df.copy()
  _df = _df.applymap(no_space_start_end)
  _df = _df.applymap(no_space_between)
  _df = _df.applymap(upper_case)
  _df = _df.applymap(words_unidecode)
  
  _df[_df==""] = np.nan

  return _df
  
  
def fuzzy_wuzzy(s_evaluado, s_buscado):
  return fuzz.partial_ratio(str(s_evaluado),s_buscado)/100 

def sequence_matcher_similarity(s_evaluado, s_buscado):
  return difflib.SequenceMatcher(None, str(s_evaluado), s_buscado).ratio()

def and_partial_ratio(s_evaluado, s_buscado):
  similarity = 1
  for s in s_buscado.split():
    similarity *= fuzz.partial_ratio(s, str(s_evaluado))/100
  return similarity

def limited_and_partial_ratio(s_evaluado, s_buscado):
  s_evaluado = str(s_evaluado)
  palabras_buscadas = s_buscado.split()
  palabras_evaluadas = s_evaluado.split()
  if len(palabras_buscadas) ==  len(palabras_evaluadas):
    similarity = 1
    for s in palabras_buscadas:
      similarity *= fuzz.partial_ratio(s, s_evaluado)/100 
    return similarity
  else:
    return difflib.SequenceMatcher(None, s_buscado, s_evaluado).ratio()
   
   
def encontrar_anhos(s): 
  if pd.notna(s):
      s_copy = s
      special_chars = [' ','/','\\','-']
      for c in special_chars:
        s_copy = s_copy.replace(c, c+c)
      expr = r"^201[0-9]$|^201[0-9][\s\/\-\\]|[\s\/\-\\]201[0-9]$|[\s\/\-\\]201[0-9][\s\/\-\\]"
      years = re.findall(expr,s_copy)
      for i  in range(len(years)):
        for c in special_chars:
          years[i] = years[i].replace(c, '')
      return years
  return None   
   
def is_year(x):
  if pd.notnull(x):
    # expr = "[\s]+201[0-9]+|^201[0-9]|[\s]201[0-9]$"
    # years = re.findall(expr,str(x))
    
    years = encontrar_anhos(x)

    return 1 if len(years)>0 else 0
  else:
    return 0
    
    
def get_year(x):
  if pd.notnull(x):
    
    # expr = "[\s]+201[0-9]+|^201[0-9]|[\s]201[0-9]$"
    # years = re.findall(expr,str(x))
    
    years = encontrar_anhos(x)
    
    years = [int(year) for year in years]
    
    return np.max(years) if len(years)>0 else 0
  else:
    return 0  
     
    
def get_variables_index(df_doc, dict_parameters, compare_function, treshold=0.85):

  dict_variables = {}

  for name_var in dict_parameters.keys():
    # variable = df_variables[name_var].values
    # variable = variable[~pd.isnull(variable)]
    variable = dict_parameters[name_var]

    probs = []
    keys = []
    coordinates = []

    for var in variable:
      matriz_prob = df_doc.applymap(lambda x: compare_function(x,var))

      coordinate = np.unravel_index(np.argmax(matriz_prob, axis=None), matriz_prob.shape)
      prob = matriz_prob.values[coordinate[0]][coordinate[1]]
      key = df_doc.values[coordinate[0]][coordinate[1]]

      probs.append(prob)
      keys.append(key)
      coordinates.append(coordinate)

    argmax = np.argmax(probs)  
    coordinate_max = coordinates[argmax]
    prob_max = probs[argmax]
    key_max = keys[argmax]
    
    if prob_max >= treshold:
      dict_variables[name_var] = [int(coordinate_max[0]),int(coordinate_max[1])]

  # with open(name_file, 'w') as file:
    # json.dump(dict_variables, file)

  return dict_variables


def quitar_vacios_dic(d):
  return {x: d[x] for x in d if pd.notna(d[x])}


def processing_values_dict(json_object_older):

  json_object = deepcopy(json_object_older)

  json_object = quitar_vacios_dic(json_object)

  for var in json_object.keys():
    value = json_object[var]
    


    if isinstance(value, str): 
    
      value = value.strip()
      value = re.sub(",","",value)
      value = re.sub("\.","",value)
      print(value)
      value = re.sub("\)$","",value)
      print(value)  
      value = re.sub("^\(","-",value)
      print(value)

      try:
        value = float(value)
      except:
        value = np.nan 
    

    json_object[var] = value

  json_object = quitar_vacios_dic(json_object)


  return json_object

  
def get_dict_vars_values(df, dict_vars, name_file="variables_indices.json"):
  
  _df = df.copy()

  pos_row = np.argmax(_df.applymap(is_year).sum(1).values)
  _df.columns = _df.iloc[pos_row]

  years_gen = [get_year(col) for col in _df.columns]
  year_gen = np.max(years_gen)

  dict_vars_values = {}

  for key in dict_vars:

    x, y = dict_vars[key]

    years = [get_year(col) for col in _df.columns[y:]]
    year = np.max(years)
    col_year_pos = np.argmax(years)
    _df_dummy = _df.iloc[:,y:] 
    value = _df_dummy.iloc[:,col_year_pos][x]
    # value = value.replace('.','')
    # value = value.replace(',','.')
    dict_vars_values[key] = value
  
  dict_vars_values["FECHA"] = int(year_gen)

  #with open(name_file, 'w') as file:
  #  json.dump(dict_vars_values, file)

  return dict_vars_values
  
  
def main():
  with open('/content/drive/My Drive/HACKATHONBBVA_2020/variable_dictionary/variables.json', 'r') as j:
    dict_parameters = json.load(j)

  path_table = "/content/drive/My Drive/HACKATHONBBVA_2020/doc_to_table/image_Doc26_variables.csv"
  path_save = "/content/drive/My Drive/HACKATHONBBVA_2020/final_values/"
  df = pd.read_csv(path_table)
  df = processing_text(df)

  dict_variables = get_variables_index(df, dict_parameters, sequence_matcher_similarity)

  dict_variables = get_dict_vars_values(df, dict_variables, name_file=path_save + path_table.split('/')[-1].split('.')[0] + '_final.json')
  # dict_variables = get_dict_vars_values(df, dict_parameters, sequence_matcher_similarity, name_file=path_save + path_table.split('/')[-1].split('.')[0] + '_final.json')


