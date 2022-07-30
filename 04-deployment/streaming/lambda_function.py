import os
import json
import boto3
import base64

import mlflow

kinesis_client = boto3.client('kinesis')
PREDICTIONS_STREAM_NAME = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')


RUN_ID = os.getenv('RUN_ID')
os.environ['AWS_PROFILE'] = 'MLOps-dev'

logged_model = f's3://mlflow-artifacts-prem/1/{RUN_ID}/artifacts/model'
model = mlflow.pyfunc.load_model(logged_model)


TEST_RUN = os.getenv('DRY_RUN', 'False') == 'True'

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features
 

def predict(features):
    preds = model.predict(features)
    return preds[0]
    
def lambda_handler(event, context):
    
    predictions_events = list()
    
    for record in event['Records']:
        encoded_data = record['kinesis']['data']
        decoded_data = base64.b64decode(encoded_data).decode('utf-8')
        ride_event = json.loads(decoded_data)
        
        
        ride = ride_event['ride']
        ride_id = ride_event['ride_id']
        
        features = prepare_features(ride)
        prediction = predict(features)
        
        prediction_event = {
            'model': 'ride_duration_prediction_model',
            'version': '1',
            'prediction': {
                'ride_duration': prediction,
                'ride_id': ride_id
            }
        }

        if not TEST_RUN:     
            response = kinesis_client.put_record(
                    StreamName=PREDICTIONS_STREAM_NAME,
                    Data=json.dumps(prediction_event),
                    PartitionKey=str(ride_id),
                ) 
        
        predictions_events.append(prediction_event)
        
        # print(f"messages from kinesis client: {response}")
            
    return {
        'predictions': predictions_events
    }
    
    

    
