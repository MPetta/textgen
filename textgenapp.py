from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import plotly.graph_objects as go
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from io import StringIO
import scipy.stats as stats
from scipy.stats import chi2_contingency

################################################################
####      ##     ##  ####  ##      ##      ##     ##  ####  ####
######  ####  ######  ##  #####  ####  ######  #####   ###  ####
######  ####     ####    ######  ####      ##     ##    ##  ####
######  ####  ######  ##  #####  ####  ##  ##  #####  #  #  ####
######  ####     ##  ####  ####  ####      ##     ##  ##    ####
################################################################

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

# text for outputs
markdown_text = '''
### Description

This app uses two pretrained large language models for the tasks of text generation and sentiment analysis.
For text generation, this app responds to user provided prompts by having GPT2 'complete' or 'append to' that input.
More information on GPT2 can be found in [this paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
. The machine generated text as well as the user input are then analyzed for sentiment and the comparison is provided below.
For this natural langauge processing task, this app uses the [RoBERTa model] (https://ai.meta.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/)
. 
'''

markdown_text_two = '''
__Is there any evidence of a relationship between class and their sentiment, at 5% significant level?__

* *class determines whether text was from a user or machine* 


H₀: whether class and their sentiment are independent, i.e. no relationship

H₁: whether class and their sentiment are dependent, i.e.  relationship

α = 0.05

'''

markdown_text_three = '''
This hypothesis testing process requires users to submit multiple prompts to generate multiple responses. Sample size is a crucial factor in any hypothesis test study design. Initially, we start with a small sample size, which users can expand as needed. The generated data is cached for the current session, and in a future update, I plan to add a power analysis tool to help users better understand the impact of sample size.
'''

# intiate plot to be updated
categories = ['compound','positive', 'neutral', 'negative']
starter = [0, 0, 0, 0]

fig = go.Figure(
    data=[
        go.Scatterpolar(r=starter, theta=categories, fill='toself', name='Machine'),
        go.Scatterpolar(r=starter, theta=categories, fill='toself', name='User'),
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

# instantiate dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
application = app.server

# create layout
app.layout = html.Div([dbc.Container([
        html.H1("Generative Text Analyzer")
        , html.H3("Does GPT2 Share Your Sentiment?")
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
        # , html.H4("This is all just to display a framework for generative text sentiment analysis. Some tuning may still be needed.")
        , html.Div(id='my-sentiment-output')
        , dcc.Graph(id='radar-plot',figure=fig)
        , dcc.Store(id='generated-data')
        , html.H3("Hypothesis Testing")
        , dcc.Markdown(children=markdown_text_two)
        , html.H5("Contingency Table")
        , html.Table(id='table') 
        , html.H5("Conclusion")
        , html.Div(id='my-test-output')
        , html.Br()
        , dcc.Markdown(children=markdown_text_three)
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

    # for reference of order -> categories = ['compound','positive', 'neutral', 'negative']
    user_fig_list = [user_compound_count, user_positive_count, user_neutral_count, user_negative_count]
    user_fig_list = [int(i) for i in user_fig_list]
    
    machine_fig_list = [machine_compound_count, machine_positive_count, machine_neutral_count, machine_negative_count]
    machine_fig_list = [int(i) for i in machine_fig_list]

    df['user_fig_list'] = str(user_fig_list)
    df['machine_fig_list'] = str(machine_fig_list)

    df['user_sent_list'] = str(user_sent_list)
    df['machine_sent_list'] = str(machine_sent_list)

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
    user_sentiment_list = [int(i) for i in user_sentiment_str.replace('[', '').replace(']', '').split(',')]

    machine_sentiment_str = dff["machine_fig_list"][0]
    machine_sentiment_list = [int(i) for i in machine_sentiment_str.replace('[', '').replace(']', '').split(',')]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=machine_sentiment_list, theta=categories, fill='toself', name='Machine'),
            go.Scatterpolar(r=user_sentiment_list, theta=categories, fill='toself', name='User'),
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

## For Debugging

@app.callback(
    Output(component_id='table', component_property='children'),
    Input(component_id='generated-data', component_property='data'),
)
def update_table(generated_data): # for debugging
    dff = pd.read_json(StringIO(generated_data), orient='split')

    user_sentiment_str = dff["user_fig_list"][0]
    user_sentiment_list = [int(i) for i in user_sentiment_str.replace('[', '').replace(']', '').split(',')]

    machine_sentiment_str = dff["machine_fig_list"][0]
    machine_sentiment_list = [int(i) for i in machine_sentiment_str.replace('[', '').replace(']', '').split(',')]

    data = [['machine', 'compound'] for i in range(machine_sentiment_list[0])] + \
        [['machine', 'positive'] for i in range(machine_sentiment_list[1])] + \
        [['machine', 'neutral'] for i in range(machine_sentiment_list[2])] + \
        [['machine', 'negative'] for i in range(machine_sentiment_list[3])] + \
        [['user', 'compound'] for i in range(user_sentiment_list[0])] + \
        [['user', 'positive'] for i in range(user_sentiment_list[1])] + \
        [['user', 'neutral'] for i in range(user_sentiment_list[2])] + \
        [['user', 'negative'] for i in range(user_sentiment_list[3])]
    dff2 = pd.DataFrame(data, columns = ['Class', 'Category'])

    data_crosstab = pd.crosstab(dff2['Class'],
                            dff2['Category'],
                           margins=True, margins_name="Total")
    crosstab_print = pd.DataFrame(data_crosstab).reset_index()

    table = dbc.Table.from_dataframe(crosstab_print, striped=True, bordered=True, hover=True)
    return table

@app.callback(
    Output(component_id='my-test-output', component_property='children'),
    Input(component_id='generated-data', component_property='data'),
)
def update_test_output(generated_data): # for debugging
    dff = pd.read_json(StringIO(generated_data), orient='split')
    user_sentiment_str = dff["user_fig_list"][0]
    user_sentiment_list = [int(i) for i in user_sentiment_str.replace('[', '').replace(']', '').split(',')]

    machine_sentiment_str = dff["machine_fig_list"][0]
    machine_sentiment_list = [int(i) for i in machine_sentiment_str.replace('[', '').replace(']', '').split(',')]

    data = [['machine', 'compound'] for i in range(machine_sentiment_list[0])] + \
        [['machine', 'positive'] for i in range(machine_sentiment_list[1])] + \
        [['machine', 'neutral'] for i in range(machine_sentiment_list[2])] + \
        [['machine', 'negative'] for i in range(machine_sentiment_list[3])] + \
        [['user', 'compound'] for i in range(user_sentiment_list[0])] + \
        [['user', 'positive'] for i in range(user_sentiment_list[1])] + \
        [['user', 'neutral'] for i in range(user_sentiment_list[2])] + \
        [['user', 'negative'] for i in range(user_sentiment_list[3])]
    dff2 = pd.DataFrame(data, columns = ['Class', 'Category'])

    data_crosstab = pd.crosstab(dff2['Class'],
                            dff2['Category'],
                           margins=True, margins_name="Total")

    stat, p, dof, expected = chi2_contingency(data_crosstab)
    alpha = 0.05
    if p <= alpha:
        chi_ouput =  "we reject the null hypothesis. Their is a relationship. -- Dependent" # Dependent (reject H0)
    else:
        chi_ouput =  "we fail to reject the null hypothesis. Their is no relationship. -- Independent" # Independent (H0 holds true)

    return html.P(f"With a p-value of:{p} - {chi_ouput}")

# run app server
if __name__ == '__main__':
    application.run()

