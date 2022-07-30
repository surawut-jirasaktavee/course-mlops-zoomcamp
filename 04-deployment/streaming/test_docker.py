import requests 

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49631818369979129679861815788598519962499637577139617794",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogOTAsCiAgICAgICAgICAgICJET0xvY2F0aW9uSUQiOiAyODUsCiAgICAgICAgICAgICJ0cmlwX2Rpc3RhbmNlIjogMy4yMgogICAgICAgIH0sCiAgICAgICAgInJpZGVfaWQiOiAxNDQKICAgIH0=",
                "approximateArrivalTimestamp": 1659013681.735
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49631818369979129679861815788598519962499637577139617794",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::551011018709:role/lambda-kinesis-role",
            "awsRegion": "us-west-1",
            "eventSourceARN": "arn:aws:kinesis:us-west-1:551011018709:stream/ride_events"
        }
    ]
}


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
response = requests.post(url, json=event)
print(response.json())
