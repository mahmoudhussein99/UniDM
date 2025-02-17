#!/usr/bin/env python
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import os
from pathlib import Path

# DATASET_PATH = os.environ.get("DATASET_PATH", Path("data/datasets").resolve())


IMPUTE_COLS = {
    f"Buy": "manufacturer",
    f"Restaurant": "city",
}

MATCH_PROD_NAME = {
    f"Amazon-Google": "Product",
    f"Beer": "Product",
    f"DBLP-ACM": "Product",
    f"DBLP-GoogleScholar": "Product",
    f"MusicBrainz20k": "Song",
    f"WDC-Computers": "Product",
    f"WDC-Shoes": "Shoes",
    f"WDC-Cameras": "Camera",
    f"WDC-Watches": "Watch",
    f"Fodors-Zagats": "Product",
    f"iTunes-Amazon": "Song",
    f"Walmart-Amazon": "Product",
    f"Synthea": "",
}

# Dropping happens before renaming
DATA2DROPCOLS = {
    f"Amazon-Google": [],
    f"Beer": ["Style", "ABV"],
    f"DBLP-ACM": [],
    f"DBLP-GoogleScholar": ["venue","year"],
    f"MusicBrainz20k": ["cluster_id","source","number","length","language"],
    f"WDC-Computers": [],
    f"WDC-Shoes": [],
    f"WDC-Watches": [],
    f"WDC-Cameras": [],
    f"Fodors-Zagats": [],
    f"iTunes-Amazon": ["CopyRight"],
    f"Walmart-Amazon": [
        "category",
        "price",
        "brand",
    ],
    f"Buy": [],
    f"Restaurant": [],
    f"Hospital": [],
    f"Adult": [],
    f"Synthea": ["des1", "des2", "d1", "d2", "d3", "d4"],
    f"benchmark-stackoverflow": [],
    f"benchmark-bing-query-logs": [],
}

DATA2COLREMAP = {
    f"Amazon-Google": {},
    f"Beer": {
        "id": "id",
        "Beer_Name": "name",
        "Brew_Factory_Name": "factory",
        "Style": "style",
        "ABV": "ABV",
    },
    f"DBLP-ACM": {},
    f"DBLP-GoogleScholar": {},
    f"MusicBrainz20k": {},
    f"WDC-Computers":{},
    f"WDC-Shoes": {},
    f"WDC-Watches": {},
    f"WDC-Cameras": {},
    f"Fodors-Zagats": {},
    f"iTunes-Amazon": {
        "id": "id",
        "Song_Name": "name",
        "Artist_Name": "artist name",
        "Album_Name": "album name",
        "Genre": "genre",
        "Price": "price",
        "CopyRight": "CopyRight",
        "Time": "time",
        "Released": "released",
    },
    f"Walmart-Amazon": {},
    f"Buy": {},
    f"Restaurant": {},
    f"Hospital": {},
    f"Adult": {},
    f"Synthea": {
        "omop": "left",
        "table": "right",
        "label": "label",
    },
    f"benchmark-stackoverflow": {},
    f"benchmark-bing-query-logs": {},
}

