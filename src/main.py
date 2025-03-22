import pandas as pd
import re
import swifter
import gzip
import glob
import logging
from lxml import etree
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_csv_and_clean(path, label):
    logging.info(f"Loading {label} notes from {path}")
    df = pd.read_csv(path)
    columns_to_drop = ["note_id", "subject_id", "hadm_id", "note_type", "note_seq", "charttime", "storetime"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    logging.info(f"Cleaning {label} text")
    df["clinical_text"] = df["text"].swifter.apply(clean_text)
    df = df.drop(columns="text")
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

    return {"pmid": pmid, "clinical_text": clinical_text}

def parse_pubmed_xml(xml_file):
    with gzip.open(xml_file, 'rb') as f:
        context = etree.iterparse(f, tag='PubmedArticle')
        for _, elem in context:
            yield extract_article_data(elem)
            elem.clear()

def load_pubmed_and_clean(pubmed_glob):
    records = []
    files = glob.glob(pubmed_glob)
    logging.info(f"Found {len(files)} PubMed XML files to parse.")
    for file in tqdm(files, desc="Parsing PubMed XML files"):
        logging.info(f"Parsing {file}")
        for record in tqdm(parse_pubmed_xml(file), desc=f"Extracting articles from {file}", leave=False):
            if record["clinical_text"]:
                records.append(record)
    df = pd.DataFrame(records)
    logging.info(f"Loaded {df.shape[0]} PubMed articles with clinical text.")
    return df

def load_data():
    discharge_path = "/data2/simon/data/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz"
    radiology_path = "/data2/simon/data/physionet.org/files/mimic-iv-note/2.2/note/radiology.csv.gz"
    pubmed_path_glob = "../data/pubmed/*.xml.gz"

    discharge = load_csv_and_clean(discharge_path, label="discharge")
    radiology = load_csv_and_clean(radiology_path, label="radiology")
    pubmed = load_pubmed_and_clean(pubmed_path_glob)

    discharge = discharge[["clinical_text"]]
    radiology = radiology[["clinical_text"]]
    pubmed = pubmed[["clinical_text"]]

    combined = pd.concat([discharge, radiology, pubmed], ignore_index=True)
    logging.info(f"Final combined dataset shape: {combined.shape}")
    return combined

# Main Script
df = load_data()

# Extra: token stats and memory usage
avg_len = df["clinical_text"].swifter.apply(lambda x: len(x.split())).mean()
mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

logging.info(f"Average token count: {avg_len:.2f}")
logging.info(f"DataFrame memory usage: {mem_mb:.2f} MB")
