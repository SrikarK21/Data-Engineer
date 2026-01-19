import pandas as pd
import pickle
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine

DB_PATH = 'sqlite:///c:/projects/DE/New folder/customer_analytics_platform/data/customer_analytics.db'
MODEL_PATH = 'c:/projects/DE/New folder/customer_analytics_platform/data/churn_model.pkl'

import pandas as pd
import pickle
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine

DB_PATH = 'sqlite:///c:/projects/DE/New folder/customer_analytics_platform/data/customer_analytics.db'
MODEL_PATH = 'c:/projects/DE/New folder/customer_analytics_platform/data/churn_model.pkl'

def load_data(engine):
    # Only transactions table exists now
    query_tx = "SELECT * FROM transactions"
    transactions = pd.read_sql(query_tx, engine)
    return transactions

def feature_engineering(transactions):
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    
    # Snapshot Date: 90 days before the very last record
    max_date = transactions['transaction_date'].max()
    snapshot_date = max_date - timedelta(days=90)
    
    print(f"Training snapshot date: {snapshot_date}")
    
    # Historical Data (Pre-Snapshot)
    hist_tx = transactions[transactions['transaction_date'] <= snapshot_date]
    
    # Calculate Features: RFM
    features = hist_tx.groupby('customer_id').agg(
        recency=('transaction_date', lambda x: (snapshot_date - x.max()).days),
        frequency=('transaction_id', 'nunique'),
        monetary=('amount', 'sum'),
        first_purchase=('transaction_date', 'min')
    ).reset_index()
    
    # Calculate Tenure
    features['tenure_days'] = (snapshot_date - features['first_purchase']).dt.days
    features = features.drop(columns=['first_purchase'])
    
    # Target Variable: Churn
    # Churn = 1 if NO transactions > snapshot_date
    future_tx = transactions[transactions['transaction_date'] > snapshot_date]
    active_customers = future_tx['customer_id'].unique()
    
    features['is_churn'] = features['customer_id'].apply(lambda x: 0 if x in active_customers else 1)
    
    # Handle negative monetary values (outliers/returns)
    features = features[features['monetary'] > 0]
    
    return features.dropna()

def train_churn_model():
    engine = create_engine(DB_PATH)
    transactions = load_data(engine)
    
    print("Engineering features...")
    df = feature_engineering(transactions)
    
    X = df.drop(columns=['is_churn', 'customer_id'])
    y = df['is_churn']
    
    print(f"Target distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    return model

if __name__ == "__main__":
    train_churn_model()
