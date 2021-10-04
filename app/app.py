import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

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
            style={'width': '100%', 'height': 300})
    ]),
    html.Button('Submit', id='submission'),
    html.Br(),
    html.Br(),
    html.Div(id='my-output'),
])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='submission', component_property='n_clicks')
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

if __name__ == '__main__':
    app.run_server(debug=True)