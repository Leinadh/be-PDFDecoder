import spacy
import pandas as pd
import os
import json

def get_currency_data(df, nlp,currency_symbols, n_first_rows):
    ls = []
    for i in range(n_first_rows):
        line_string = "".join([value for value in data.iloc[i,:].values.astype(str) if value!='nan'])
        doc = nlp(line_string)
        identity = [{ent.text, ent.label_} for ent in doc.ents]
        for ent in doc.ents:
            if ent.label_ == 'MONEY':
                ls.append(ent.text)
        if any(ext in line_string for ext in list(currency_symbols.keys())):
            ls.append(ent.text)
    return ls

if __name__ == "__main__":
    n_first_rows = 6
    nlp = spacy.load(r'D:\Competencias\BBVA\aws\be-PDFDecoder\utils\helper_functions\model')
    data = pd.read_csv(r'D:\Competencias\BBVA\aws\be-PDFDecoder\utils\flujo_completo\output_csv').iloc[:n_first_rows,:]
    with open(r'D:\Competencias\BBVA\aws\be-PDFDecoder\utils\currency_symbols.json', 'r') as file:
        currency_symbols = json.load(file)
    ls = get_currency_data(data, nlp,currency_symbols, n_first_rows)
    print (ls)