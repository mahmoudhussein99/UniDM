import spacy
import numpy as np
import pandas as pd
# PARSING_OPTIONS = ['Yes\n','No\n', "Can't decide\n"]
CONFIRM = "Yes\n"
DENY = "No\n"
ABSTAIN = "Can't decide\n"
NAN_TOKEN="nan"
IGNORED_COLUMNS= ["ltable_id","rtable_id","id","id_A","id_B","label_str","label","serialized_A","serialized_B"]

nlp = spacy.load("en_core_web_lg")

def are_texts_similar(text1, text2,label, threshold=0.8):
    if pd.isna(text1) or pd.isna(text2) :
        return ABSTAIN
    print(f"text1 is:: {text1}")
    print(f"text2 is:: {text2}")
    doc1, doc2 = nlp(text1), nlp(text2)
    similarity = doc1.similarity(doc2)  # Compute similarity score
    print(f"similarity is {similarity}")
    # use spacy to get threshold and return always yes if they are matching already
    return CONFIRM if similarity >= threshold or label == CONFIRM else DENY


def get_chain_of_thoughts_template(row,columns_list, prod_name):
    columns_for_matching=   {c.split("_")[0] for c in columns_list if not (c in IGNORED_COLUMNS) }
    # print("ROW IS::")
    # print(row)
    label = row["label_str"]
    chain_of_thoughts=f"You are an expert in {prod_name} entity resolution. Given two product data, analyze their attributes step by step and determine if they refer to the same product.\n"
    are_these_attributes_similar= f"Are %s similar for such {prod_name}s? Yes or No. %s"
    for col in columns_for_matching:
        if pd.isna(row[col+"_A"]) or pd.isna(row[col+"_B"]) :
            continue
        chain_of_thoughts+= f"regarding '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
        chain_of_thoughts += are_these_attributes_similar % (col, are_texts_similar(row[col+"_A"],row[col+"_B"],label=label) )
    return chain_of_thoughts

def get_chain_of_thoughts_template_eval(row,columns_list, prod_name):
    columns_for_matching=   {c.split("_")[0] for c in columns_list if not (c in IGNORED_COLUMNS) }
    # print("ROW IS::")
    # print(row)
    # label = row["label_str"]
    chain_of_thoughts=f"Now, Given two {prod_name} data, analyze their attributes step by step and determine if they refer to the same product.\n"
    are_these_attributes_similar= f"Are %s similar for such {prod_name}s? Yes or No. %s"
    for col in columns_for_matching:
        if pd.isna(row[col+"_A"]) or pd.isna(row[col+"_B"]) :
            continue
        chain_of_thoughts+= f"regarding '{col}':\n Product A's {col} is {str(row[col+'_A'])}\n Product B's {col} is {str(row[col+'_B'])}\n"
        chain_of_thoughts += are_these_attributes_similar % (col, are_texts_similar(row[col+"_A"],row[col+"_B"],label=DENY) )
    return chain_of_thoughts