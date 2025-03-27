import os
import glob
import gzip
import logging
import re
import pandas as pd
import swifter
from lxml import etree
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset
from src.utils import clean_text

def load_csv_and_clean(path, label, chunk_size=100000):
    logging.info(f"Loading {label} notes from {path}")
    dfs = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        columns_to_drop = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]
        chunk = chunk.drop(columns=[col for col in columns_to_drop if col in chunk.columns])
        chunk["clinical_text"] = chunk["text"].swifter.apply(clean_text)
        chunk = chunk.drop(columns="text")
        dfs.append(chunk)
    df = pd.concat(dfs)
    logging.info(f"{label} loaded with shape {df.shape}")
    return df

def extract_article_data(article):
    def xpath_text(elem, path):
        found = elem.find(path)
        return found.text.strip() if found is not None and found.text else None

    pmid = xpath_text(article, ".//PMID")
    title = xpath_text(article, ".//ArticleTitle")
    abstract_parts = article.findall(".//Abstract/AbstractText")
    abstract = " ".join([a.text.strip() for a in abstract_parts if a is not None and a.text]) if abstract_parts else None

    if title and abstract:
        clinical_text = f"{title} {abstract}"
    elif title:
        clinical_text = title
    elif abstract:
        clinical_text = abstract
    else:
        clinical_text = None

    return {"clinical_text": clinical_text}

def parse_pubmed_xml_file(file):
    logging.info(f"Parsing {file}")
    records = []
    with open(file, 'rb') as f:
        context = etree.iterparse(f, tag='PubmedArticle')
        for _, elem in context:
            record = extract_article_data(elem)
            if record["clinical_text"]:
                records.append(record)
            elem.clear()
    return records

def load_pubmed_and_clean(pubmed_glob, num_workers=8):
    files = glob.glob(pubmed_glob)
    logging.info(f"Found {len(files)} PubMed XML files to parse.")
    records = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(parse_pubmed_xml_file, file) for file in files]
        for future in tqdm(futures, desc="Processing PubMed files"):
            records.extend(future.result())
    df = pd.DataFrame(records)
    logging.info(f"Loaded {df.shape[0]} PubMed articles with clinical text.")
    return df

def load_data():
    discharge_path = "./data/mimic-note/discharge.csv.gz"
    radiology_path = "./data/mimic-note/radiology.csv.gz"
    pubmed_path_glob = "./data/pubmed/*.xml"
    
    discharge = load_csv_and_clean(discharge_path, label="discharge")
    radiology = load_csv_and_clean(radiology_path, label="radiology")
    pubmed = load_pubmed_and_clean(pubmed_path_glob)
    
    icd_code = pd.read_csv("./data/coded/icd_codes.csv")
    procedure_code = pd.read_csv("./data/coded/icd_procedures.csv")
    hcpcs_code = pd.read_csv("./data/coded/hcpcs_codes.csv")
    icd_code["clinical_text"] = icd_code["text"]
    procedure_code["clinical_text"] = procedure_code["text"]
    hcpcs_code["clinical_text"] = hcpcs_code["text"]
    
    icd_code = icd_code[["clinical_text"]]
    procedure_code = procedure_code[["clinical_text"]]
    hcpcs_code = hcpcs_code[["clinical_text"]]
    
    discharge = discharge[["clinical_text"]]
    radiology = radiology[["clinical_text"]]
    pubmed = pubmed[["clinical_text"]]
    
    combined = pd.concat([discharge, radiology, pubmed, icd_code, procedure_code, hcpcs_code], ignore_index=True)
    logging.info(f"Final combined dataset shape: {combined.shape}")
    return combined

def create_extended_context_dataset(dataset, max_length=8192):
    def combine_examples(examples, target_length):
        combined_texts = []
        current_text = ""
        for text in examples:
            if len(current_text) + len(text) <= target_length:
                current_text += text + " "
            else:
                if current_text:
                    combined_texts.append(current_text.strip())
                current_text = text + " "
        if current_text:
            combined_texts.append(current_text.strip())
        return combined_texts

    all_texts = dataset["clinical_text"]
    long_texts = combine_examples(all_texts, max_length)
    return Dataset.from_dict({"clinical_text": long_texts})

def upsample_quality_sources(dataset, quality_indices, upsample_factor=2.0):
    upsampled_data = []
    for idx, example in enumerate(dataset):
        upsampled_data.append(example)
        if idx in quality_indices:
            for _ in range(int(upsample_factor - 1)):
                upsampled_data.append(example)
    return Dataset.from_list(upsampled_data)
