from bs4 import BeautifulSoup
import requests
import os
from dotenv import load_dotenv
import time
import pandas as pd

from langchain.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

# from utils.utils import load_vectorstore

load_dotenv()

shop_url = "https://privatus-testing-store.myshopify.com"
api_version = "2023-07"

# Helpers


def get_all_products(shop_url, api_version):
    all_products = []
    url = f"{shop_url}/admin/api/{api_version}/products.json"
    headers = {"X-Shopify-Access-Token": os.getenv("SHOPIFY_API_KEY")}
    params = {"limit": 250}
    response = requests.get(url, headers=headers, params=params)
    all_products.extend(response.json()["products"])
    try:
        while response.links["next"]:
            response = requests.get(
                response.links["next"]["url"], headers=headers)
            all_products.extend(response.json()["products"])
            time.sleep(2)
    except KeyError:
        return all_products


def clean_html_tags(row):
    soup = BeautifulSoup(row["body_html"], "html.parser")
    text = soup.get_text()
    row["body_html"] = text
    return row


def get_img_src(row):
    all_images = []
    for image in row["images"]:
        all_images.append(image["src"])
    row["images_list"] = all_images
    return row


def create_expandend_description(row):
    if row["body_html"] == "" and row["tags"] == "":
        row["expanded_description"] = row["title"]
    elif row["body_html"] == "" and row["tags"] != "":
        row["expanded_description"] = "Title: " + \
            row['title'] + " Tags: " + row['tags']
    elif row["body_html"] != "" and row["tags"] == "":
        row["expanded_description"] = "Title: " + \
            row['title'] + " Description: " + row["body_html"]
    else:
        row["expanded_description"] = "Title: " + row['title'] + \
            " Description: " + row["body_html"] + " Tags: " + row['tags']
    return row


def df_preprocessing(df):
    df = df[df["status"] == "active"]
    df.fillna("", inplace=True)
    df = df.apply(lambda row: get_img_src(row), axis=1)
    df = df.apply(lambda row: create_expandend_description(row), axis=1)
    df = df.apply(lambda row: clean_html_tags(row), axis=1)
    df = df.rename(columns={"body_html": "description"})
    df = df[["id", "title", "handle", "description",
             "expanded_description", "images_list"]]
    return df


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["id"] = record.get("id")
    metadata["title"] = record.get("title")
    metadata["tags"] = record.get("tags")
    metadata["images_list"] = record.get("images_list")
    metadata["handle"] = record.get("handle")
    return metadata


def count_tokens(splitter, text):
    return splitter.count_tokens(text=text)


def generate_products_json(request):
    # all_products = get_all_products(shop_url, api_version)
    # product_df = pd.DataFrame(all_products)
    new_product = request.json
    print('new product =', new_product)
    product_df = pd.DataFrame([new_product])
    cleaned_df = df_preprocessing(product_df)
    cleaned_df.to_csv("products.csv", index=False)
    products_json = cleaned_df.to_json(orient="records")

    with open("products.json", "w") as f:
        f.write(products_json)

    return 'success'
