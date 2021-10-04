import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from google.api_core.client_options import ClientOptions
from google.cloud import automl_v1
import json

project_id = "sentiment-analysis-01"
model_id = "TST1490117531589935104"
client = automl_v1.AutoMlClient()
# Get the full path of the model.
model_full_id = client.model_path(project_id, "us-central1", model_id)
model = client.get_model(name=model_full_id)
model_name = model.name

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Tweet sentiment analysis.'),

    html.Div(children='''
        Trained on Airline tweets from this dataset:
        https://www.kaggle.com/crowdflower/twitter-airline-sentiment.
        Using GCP AutoML Sentiment Analysis, tweets are categorized into 
        negative, neutral or positive.
    '''),
    html.Br(),
    html.Div([
        "Input: ",
        dcc.Textarea(id='my-input', 
            value='Enter a tweet here and click submit.', 
            style={'width': '100%', 'height': 150})
    ]),
    html.Button('Submit', id='submission'),
    html.Br(),
    html.Br(),
    html.Div(id='my-output'),
])

def get_prediction(payload, model_name):
    options = ClientOptions(api_endpoint='automl.googleapis.com')
    prediction_client = (
        automl_v1.PredictionServiceClient(client_options=options))

    request = prediction_client.predict(name=model_name, payload=payload)

    return request  # waits until request is returned

@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='submission', component_property='n_clicks')],
    [State('my-input', 'value')]
)
def update_output_div(n_clicks, input_value):
    payload = {'text_snippet': {'content': input_value, 
        'mime_type': 'text/plain'} }
    prediction = get_prediction(payload, model_name)
    sent = prediction.payload[0].text_sentiment.sentiment
    if sent == 0:
        return 'Output: {}'.format("Negative")
    elif sent == 1:
        return 'Output: {}'.format("Neutral")
    elif sent == 2:
        return 'Output: {}'.format("Positive")
    else:
        return 'Output: {}'.format("Error")

if __name__ == '__main__':
    app.run_server(debug=True)