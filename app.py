from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
#nltk.download('vader_lexicon')
from io import StringIO

# set up for tokeniezer and pretrained language model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# set up pipeline for sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()


# initiate variables to be updated
gen_text_list = []
exv = 0

user_sent_list = []
machine_sent_list = []

sentiment_data = ''

new_machine_sentiment = {}
new_user_sentiment = {}

markdown_text = '''
### Description

Messing about with text gen...  This app uses two pretrained large language models for the tasks of text generation and sentiment analysis.
For text generation, this app responds to user provided prompts by having GPT2 'complete' or 'append to' that input.
More information on GPT2 can be found in [this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
. The machine generated text as well as the user input are then analyzed for sentiment and the comparison is provided below.
For this natural langauge processing task, this app uses the [RoBERTa model] (https://ai.meta.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)
. 
'''

# intiate plot to be updated
categories = ['compound','positive', 'neutral', 'negative']
user_starter = [10, 8, 2, 6]
machine_starter = [3, 12, 20,15]
fig = go.Figure(
    data=[
        go.Scatterpolar(r=user_starter, theta=categories, fill='toself', name='Machine'),
        go.Scatterpolar(r=machine_starter, theta=categories, fill='toself', name='User'),
    ],
    layout=go.Layout(
        title=go.layout.Title(text='<b>Sentiment comparison<b>'),
        polar={'radialaxis': {'visible': True, 'showticklabels': False}},
        showlegend=True,
        autosize=True,
        title_x=0.5
    )
)

# create an input field
def textareas():
    return html.Div([ 
            dbc.Textarea(id='my-input'
                         , size="lg"
                         , placeholder="Enter text for auto completion")
            , dbc.Button("Submit"
                         , id="gen-button"
                         , className="me-2"
                         , n_clicks=0)
            ])

#instantiate dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

#create layout
app.layout = html.Div([dbc.Container([
        html.H1("Generative Text Analyzer")
        , html.H2("Does GPT2 Share your Sentiment")
        , html.Br()
        , dcc.Markdown(children=markdown_text)
        , html.H3("Enter a prompt.")
        , textareas()
        , html.Br()
        , html.Br()
        , html.H3("Generated text will appear below.")
        , html.Div(id='my-output')
        , dbc.Button("Clear", id="clear-button", className="me-2", n_clicks=0)
        , html.Br()
        , html.Br()
        , html.H3("Sentiment Analysis")
        , html.Div(id='my-sentiment-output')
        , dcc.Graph(id='radar-plot',figure=fig)
        , dcc.Store(id='generated-data')
#         , html.Table(id='table') # for debugging
#         , html.Div(id='my-test-output') # for debugging
   ])
  ])

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output(component_id='generated-data', component_property='data'),
    Input(component_id='gen-button', component_property='n_clicks'),
    Input(component_id='clear-button', component_property='n_clicks'),
    State(component_id='my-input', component_property='value')
)
def update_output_div(gen, cl, input_value):
    gen_text = ""
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global gen_text_list
    global exv

    if 'gen-button' in changed_id:
        if input_value is None or input_value == "":
            input_value = ""
            gen_text = ""
        else:
            input_ids = tokenizer(text_target=input_value, return_tensors="pt").input_ids
            gen_tokens = model.generate(
                input_ids,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.9,
                max_length=200,
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            gen_text_list.append(gen_text)
    if 'clear-button' in changed_id:
        gen_text = ''
        exv = 0
        gen_text_list = []

    df = pd.DataFrame([gen_text], columns=['gen_text'])
    df['user_text'] = input_value

    # create sentiment analysis
    new_machine_sentiment = sentiment_analyzer.polarity_scores(str(gen_text))
    new_user_sentiment = sentiment_analyzer.polarity_scores(str(input_value))
    df['machine_sentiment'] = str(new_machine_sentiment)
    df['user_sentiment'] = str(new_user_sentiment)
    

    # process data to be passed to graph

    # add new values from user input and machine output
    user_sent_list.append(max(new_user_sentiment, key=new_user_sentiment.get))
    machine_sent_list.append(max(new_machine_sentiment, key=new_machine_sentiment.get))

    # create seperate lists by category
    user_neutral_count = user_sent_list.count('neu')
    machine_neutral_count = machine_sent_list.count('neu')

    user_negative_count = user_sent_list.count('neg')
    machine_negative_count = machine_sent_list.count('neg')

    user_positive_count = user_sent_list.count('pos')
    machine_positive_count = machine_sent_list.count('pos')

    user_compound_count = user_sent_list.count('compound')
    machine_compound_count = machine_sent_list.count('compound')

    # create proportion lists for sending to fig
    n = len(user_sent_list)

    user_prop_neu = user_neutral_count/n*100
    user_prop_neg = user_negative_count/n*100
    user_prop_pos = user_positive_count/n*100
    user_prop_comp = user_compound_count/n*100

    machine_prop_neu = machine_neutral_count/n*100
    machine_prop_neg = machine_negative_count/n*100
    machine_prop_pos = machine_positive_count/n*100
    machine_prop_comp = machine_compound_count/n*100

    # for reference of order -> categories = ['positive', 'neutral', 'negative']
    user_fig_list = [user_prop_comp, user_prop_pos, user_prop_neu, user_prop_neg]
    user_fig_list = [int(i) for i in user_fig_list]
    
    machine_fig_list = [machine_prop_comp, machine_prop_pos, machine_prop_neu, machine_prop_neg]
    machine_fig_list = [int(i) for i in machine_fig_list]

    df['user_fig_list'] = str(user_fig_list)
    df['machine_fig_list'] = str(machine_fig_list)

    # convert to json per requirement to pass to other callbacks
    generated_data = df.to_json(date_format='iso', orient='split')

    # output options
    output_one = html.P(gen_text)


    return output_one, generated_data

@app.callback(
    Output(component_id='my-sentiment-output', component_property='children'),
    Input(component_id='generated-data', component_property='data'),
)
def update_sentiment_output_div(generated_data):
    dff = pd.read_json(StringIO(generated_data), orient='split')
    user_sent_score = dff["user_sentiment"][0]
    machine_sent_score = dff["machine_sentiment"][0]
    output_two = html.P(f"Sentiment of Generated Text:{machine_sent_score}")
    output_three = html.P(f"Sentiment of User Text:{user_sent_score}")
    return output_two, output_three


@app.callback(
    Output(component_id='radar-plot', component_property='figure'),
    Input(component_id='generated-data', component_property='data'),
)
def update_graph(generated_data):
    dff = pd.read_json(StringIO(generated_data), orient='split')
    user_sentiment_str = dff["user_fig_list"][0]
    user_sentiment_list = list(user_sentiment_str)
    machine_sentiment_str = dff["machine_fig_list"][0]
    machine_sentiment_list = list(machine_sentiment_str)

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=user_sentiment_list, theta=categories, fill='toself', name='Machine'),
            go.Scatterpolar(r=machine_sentiment_list, theta=categories, fill='toself', name='User'),
        ],
        layout=go.Layout(
            title=go.layout.Title(text='<b>Sentiment Comparison<b>'),
            polar={'radialaxis': {'visible': True, 'showticklabels': False}},
            showlegend=True,
            autosize=True,
            title_x=0.5
        )
    )
    return fig

### For Debugging

# @app.callback(
#     Output(component_id='table', component_property='children'),
#     Input(component_id='generated-data', component_property='data'),
# )
# def update_table(generated_data): # for debugging
#     dff = pd.read_json(generated_data, orient='split')
#     table = dbc.Table.from_dataframe(dff, striped=True, bordered=True, hover=True)
#     return table

# @app.callback(
#     Output(component_id='my-test-output', component_property='children'),
#     Input(component_id='generated-data', component_property='data'),
# )
# def update_test_output(generated_data): # for debugging
#     dff = pd.read_json(generated_data, orient='split')
#     categories = ['positive', 'neutral', 'negative']
#     user_sentiment_for_print = dff["user_fig_list"][0]
#     user_sentiment_r = np.array(user_sentiment_for_print)
#     # user_sentiment_values = pd.Series(user_sentiment_values)

#     machine_sentiment_for_print = dff["machine_fig_list"][0]
#     machine_sentiment_r = np.array(machine_sentiment_for_print)

#     return html.P(machine_sentiment_r)

#run app server
if __name__ == '__main__':
    app.run_server(debug=False)
