import pandas as pd
from model_training import label_encoding
from pandas.testing import assert_frame_equal


def test_make_label_encoding():
    
    '''
    Makes the unit test with the function 'label_encoding' from 'model_training.py'
    return None if the expected result is equivalent to the actual result,
    return difference in otherwise cases
    '''

    customer = pd.DataFrame(
        {"Account length": 117,
        "International plan": "No",
        "Voice mail plan": "No",
        "Number vmail messages": 0,
        "Total day minutes": 0,
        "Total day calls": 97,
        "Total eve minutes": 351.6,
        "Total eve calls": 80,
        "Total night minutes": 215.8,
        "Total night calls": 90,
        "Total intl minutes": 8.7,
        "Total intl calls": 4,
        "Customer service calls": 4
        },
        index=[0]
        )

    actual_result = label_encoding(customer)

    expected_result = pd.DataFrame(
        {"Account length": 117,
        "International plan": 0,
        "Voice mail plan": 0,
        "Number vmail messages": 0,
        "Total day minutes": 0,
        "Total day calls": 97,
        "Total eve minutes": 351.6,
        "Total eve calls": 80,
        "Total night minutes": 215.8,
        "Total night calls": 90,
        "Total intl minutes": 8.7,
        "Total intl calls": 4,
        "Customer service calls": 4
        },
        index=[0]
    )
    assertion = assert_frame_equal(actual_result, expected_result)
    assert assertion is None