from datetime import datetime
from time import sleep

import json
import uuid

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from pathlib import Path

import pandas as pd
import pyarrow as pa

table = pd.read_csv("./evidently_service/datasets/churn_test_dataset.csv")
table = pa.Table.from_pandas(table, preserve_index=True)
data = table.to_pylist()

class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


report_path = Path(f"./churn_report")
report_path.mkdir(parents=True, exist_ok=True)

with open(f"{report_path}/customer_churn_result.csv", "w") as f_result:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=1)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    for row in data:
        row['customer_ID'] = str(uuid.uuid4())
        # Send the data to the prediction service with a few seconds pause
        f_result.write(f"{row['customer_ID']}, {row['Churn']}\n") 
        response = requests.post("http://127.0.0.1:9696/predict",
                                headers={"Content-Type": "application/json"},
                                data=json.dumps(row, cls=DateTimeEncoder)).json()
        print(f"prediction: {response['customer_churn_prediction']}")
        sleep(3)
        

