"""
This script is for transforming original text datasets into
one that matches the specifications required for gcp automl sentiment analysis:
https://cloud.google.com/natural-language/automl/docs/prepare

It should follow the form:
set (optional) | content | category

set is used to clarify if you want a row used for TRAIN, TEST or VALIDATION
If this optional column is not included autoML will split it into 80% 10% 10%

Data source for this project:
https://www.kaggle.com/crowdflower/twitter-airline-sentiment
"""
# %%
import pandas as pd
from pandas._libs import interval

RAW_PATH = "./data/tweets.csv"
CLEAN_PATH = "./data/clean_tweets.csv"

class AutoMLwrangler:
    def __init__(self) -> None:
        self.raw_data = None
        self.clean_data = None

    def import_data(self, *args, **kwargs) -> int:
        """ 
        A wrapper for pandas read_csv, provide a path to a csv and it will
        store the data as a dataframe in the AutoMLwrangler object

        Args: 
            All args/kwargs read_csv takes
        returns:
            (int): Number of rows in the dataframe

        """
        self.raw_data = pd.read_csv(*args, **kwargs)
        return self.raw_data.shape[0]

    def set_columns(self, content, category, set=False):
        if set is False:
            self.clean_data = self.raw_data[[content, category]]

    def clean_columns(self):
        pass

    def export_data(self, path):
        pass

wrangler = AutoMLwrangler()
print(wrangler.import_data(RAW_PATH))
wrangler.set_columns("text", "airline_sentiment")

# %%
