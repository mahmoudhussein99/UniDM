import json
import re
import pandas as pd
# Path to your WDC-Computers dataset file
dataset_path = 'path/to/wdc-computers-dataset.jsonl'
dataset_path = {}
dataset_path['Cameras'] = '/scratch/mhussein/UniDM/dataset/datasets/entity_matching/structured/WDC-Cameras/test.csv'
dataset_path['Computers'] = '/scratch/mhussein/UniDM/dataset/datasets/entity_matching/structured/WDC-Computers/test.csv'
dataset_path['Shoes'] = '/scratch/mhussein/UniDM/dataset/datasets/entity_matching/structured/WDC-Shoes/test.csv'
dataset_path['Watches'] = '/scratch/mhussein/UniDM/dataset/datasets/entity_matching/structured/WDC-Watches/test.csv'
# Regex pattern to detect language tags at the end of the text (e.g., @en, @de)
lang_tag_pattern = re.compile(r'@([a-z]{2})(\s|$)', re.IGNORECASE)

for prod in dataset_path.keys():
    # Counters for analysis
    # Read the dataset using pandas
    
    df = pd.read_csv(dataset_path[prod])

    # Counters for analysis
    total_entries = len(df)

    # Title Analysis
    title_A_total = df['title_A'].notna().sum()
    title_B_total = df['title_B'].notna().sum()

    title_A_with_lang_tag = df['title_A'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()
    title_B_with_lang_tag = df['title_B'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()

    # Brand Analysis
    brand_A_total = df['brand_A'].notna().sum()
    brand_B_total = df['brand_B'].notna().sum()

    brand_A_with_lang_tag = df['brand_A'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()
    brand_B_with_lang_tag = df['brand_B'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()

    # Description Analysis
    description_A_total = df['description_A'].notna().sum()
    description_B_total = df['description_B'].notna().sum()

    description_A_with_lang_tag = df['description_A'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()
    description_B_with_lang_tag = df['description_B'].fillna('').apply(lambda x: bool(lang_tag_pattern.search(x))).sum()

    # Calculate percentages
    title_A_lang_tag_percent = (title_A_with_lang_tag / title_A_total) * 100 if title_A_total else 0
    title_B_lang_tag_percent = (title_B_with_lang_tag / title_B_total) * 100 if title_B_total else 0

    description_A_lang_tag_percent = (description_A_with_lang_tag / description_A_total) * 100 if description_A_total else 0
    description_B_lang_tag_percent = (description_B_with_lang_tag / description_B_total) * 100 if description_B_total else 0

    brand_A_lang_tag_percent = (brand_A_with_lang_tag / brand_A_total) * 100 if brand_A_total else 0
    brand_B_lang_tag_percent = (brand_B_with_lang_tag / brand_B_total) * 100 if brand_B_total else 0
    
    # Display results
    print(f"{prod}-Total entries: {total_entries}")
    print(f"{prod}-Title_A - Total: {title_A_total}, With Lang Tag: {title_A_with_lang_tag} ({title_A_lang_tag_percent:.2f}%)")
    print(f"{prod}-Title_B - Total: {title_B_total}, With Lang Tag: {title_B_with_lang_tag} ({title_B_lang_tag_percent:.2f}%)")
    print(f"{prod}-Description_A - Total: {description_A_total}, With Lang Tag: {description_A_with_lang_tag} ({description_A_lang_tag_percent:.2f}%)")
    print(f"{prod}-Description_B - Total: {description_B_total}, With Lang Tag: {description_B_with_lang_tag} ({description_B_lang_tag_percent:.2f}%)")
    print(f"{prod}-Brand_A - Total: {brand_A_total}, With Lang Tag: {brand_A_with_lang_tag} ({brand_A_lang_tag_percent:.2f}%)")
    print(f"{prod}-Brand_B - Total: {brand_B_total}, With Lang Tag: {brand_B_with_lang_tag} ({brand_B_lang_tag_percent:.2f}%)")