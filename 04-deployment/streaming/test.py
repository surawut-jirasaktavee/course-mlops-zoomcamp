import lambda_function

event = {
    "Records": [
        {
            "kinesis": {
                "kinesisSchemaVersion": "1.0",
                "partitionKey": "1",
                "sequenceNumber": "49631818369979129679861768006405126900808089115104903170",
                "data": "ewogICAgICAgICJyaWRlIjogewogICAgICAgICAgICAiUFVMb2NhdGlvbklEIjogMTMwLAogICAgICAgICAgICAiRE9Mb2NhdGlvbklEIjogMjA1LAogICAgICAgICAgICAidHJpcF9kaXN0YW5jZSI6IDMuNjYKICAgICAgICB9LAogICAgICAgICJyaWRlX2lkIjogMjU2CiAgICB9",
                "approximateArrivalTimestamp": 1659013681.735
            },
            "eventSource": "aws:kinesis",
            "eventVersion": "1.0",
            "eventID": "shardId-000000000000:49631818369979129679861768006405126900808089115104903170",
            "eventName": "aws:kinesis:record",
            "invokeIdentityArn": "arn:aws:iam::551011018709:role/lambda-kinesis-role",
            "awsRegion": "us-west-1",
            "eventSourceARN": "arn:aws:kinesis:us-west-1:551011018709:stream/ride_events"
        }
    ]
}

result = lambda_function.lambda_handler(event, None)
print(result)
