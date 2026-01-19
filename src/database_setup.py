import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
import requests

# Configuration - Use relative paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

DB_PATH = f'sqlite:///{os.path.join(DATA_DIR, "customer_analytics.db")}'
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
LOCAL_FILE = os.path.join(DATA_DIR, 'online_retail.xlsx')

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String) # InvoiceNo
    customer_id = Column(Integer)   # CustomerID
    stock_code = Column(String)     # StockCode
    description = Column(String)    # Description
    quantity = Column(Integer)      # Quantity
    transaction_date = Column(DateTime) # InvoiceDate
    unit_price = Column(Float)      # UnitPrice
    amount = Column(Float)          # Calculated Total Amount
    country = Column(String)        # Country

def init_db():
    engine = create_engine(DB_PATH)
    Base.metadata.drop_all(engine) # Drop old schema to align with new data
    Base.metadata.create_all(engine)
    return engine

def download_data():
    if os.path.exists(LOCAL_FILE):
        print(f"File already exists at {LOCAL_FILE}")
        return

    print(f"Downloading data from {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(LOCAL_FILE, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download data: {e}")
        # Fallback or exit? For now, let's assume it works or user provides file.
        raise e

def process_and_ingest(engine):
    print("Loading Excel file (this might take a moment)...")
    # Read Excel - UCI dataset is usually in the first sheet
    df = pd.read_excel(LOCAL_FILE)
    
    print(f"Raw data shape: {df.shape}")
    
    # Cleaning Pipeline
    print("Cleaning data...")
    # 1. Drop rows with no CustomerID (we need this for customer analytics)
    df = df.dropna(subset=['CustomerID'])
    
    # 2. Remove Cancelled transactions (InvoiceNo starts with 'C')
    # Converting to string ensuring we catch the pattern
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]
    
    # 3. Ensure Quantity is positive (redundant check after removing 'C' but good practice)
    df = df[df['Quantity'] > 0]
    
    # 4. Calculate Total Amount
    df['amount'] = df['Quantity'] * df['UnitPrice']
    
    # 5. Rename columns to match schema
    df = df.rename(columns={
        'InvoiceNo': 'transaction_id',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'transaction_date',
        'UnitPrice': 'unit_price',
        'CustomerID': 'customer_id',
        'Country': 'country'
    })
    
    print(f"Cleaned data shape: {df.shape}")
    
    # Ingest to SQLite
    print("Ingesting into SQLite...")
    # Using 'append' to bulk insert. Index=False as we have autoincrement ID.
    df.to_sql('transactions', engine, if_exists='append', index=False)
    print("Ingestion complete.")

if __name__ == "__main__":
    download_data()
    db_engine = init_db()
    process_and_ingest(db_engine)
