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

    def set_columns(self, content, category, set=False, set_col=None) -> None:
        """ 
        selects the columns according to the schema:
        set (optional) | content | category

        Input the names of the content and category and the subset dataframe 
        will be stored. If set is true and a set column is provided it will
        output with a custom TEST, TRAIN, VALIDATION assignment column.

        Args: 
            content(str): Name of the column containing content
            category(str): Name of the column containing categories
            set(bool): Whether or not to include a set column
            set_col(pandas.core.series.Series): A series containing TEST, TRAIN,
                VALIDATION assignments in each row, needs to be same length as 
                dataframe.

        Returns:
            None
        """
        if set is False:
            self.clean_data = self.raw_data[[content, category]]
        else:
            #TODO: if set is true, merge the set_col series with the clean_data
            pass

    def clean_columns(self) -> int:
        """
        Removes rows containing:
            - Unicode characters in labels. 
            - spaces and non-alphanumeric characters in labels.
            - Empty lines.
        
        Returns:
            (int): Length of clean_data after removing rows
        """
        return self.clean_data.shape[0]

    def export_data(self, *args, **kwargs) -> None:
        """ 
        A wrapper for pandas to_csv, provide a path to a csv and it will
        store the clean data without headers or an index

        Args: 
            All args/kwargs to_csv takes
        """
        self.clean_data.to_csv(*args, **kwargs, index=False, header=False)

wrangler = AutoMLwrangler()
print(wrangler.import_data(RAW_PATH))
wrangler.set_columns("text", "airline_sentiment")
wrangler.export_data(CLEAN_PATH)

# %%

