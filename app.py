import json
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from groq import Groq
from sklearn.linear_model import LinearRegression
import numpy as np
import datetime

# Initialize Groq client
client = Groq()

# Define stock analysis functions
def get_stock_price(ticker):
    stock_data = yf.Ticker(ticker).history(period='1d')
    return str(stock_data.iloc[-1].Close)

def calculate_SMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.rolling(window=window).mean().iloc[-1])

def calculate_EMA(ticker, window):
    data = yf.Ticker(ticker).history(period='1y').Close
    return str(data.ewm(span=window, adjust=False).mean().iloc[-1])

def calculate_RSI(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14-1, adjust=False).mean()
    ema_down = down.ewm(com=14-1, adjust=False).mean()
    rs = ema_up / ema_down
    return str(100 - (100 / (1 + rs)).iloc[-1])

def calculate_MACD(ticker):
    data = yf.Ticker(ticker).history(period='1y').Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    MACD_histogram = MACD - signal
    return f'{MACD[-1]}, {signal[-1]}, {MACD_histogram[-1]}'

def plot_stock_price(ticker):
    data = yf.Ticker(ticker).history(period='1y')
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close)
    plt.title(f'{ticker} Stock Price Over Last Year')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.grid(True)
    plt.savefig('stock.png')
    plt.close()

def predict_stock_price(ticker, days_ahead):
    data = yf.Ticker(ticker).history(period='1y')
    data = data.reset_index()
    data['Date'] = (data['Date'] - data['Date'].min()).dt.days
    X = data[['Date']].values
    y = data['Close'].values

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array([[X[-1, 0] + days_ahead]])
    prediction = model.predict(future_days)

    return str(prediction[0])

# Define available functions
available_functions = {
    'get_stock_price': get_stock_price,
    'calculate_SMA': calculate_SMA,
    'calculate_RSI': calculate_RSI,
    'calculate_EMA': calculate_EMA,
    'calculate_MACD': calculate_MACD,
    'plot_stock_price': plot_stock_price,
    'predict_stock_price': predict_stock_price
}

# Initialize Streamlit session state
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

st.title('Stock Analysis Chatbot Assistant')

user_input = st.text_input('Your input:')

if user_input:
    try:
        st.session_state['messages'].append({'role': 'user', 'content': user_input})

        # Call Groq API
        response = client.chat.completions.create(
            messages=st.session_state['messages'],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False
        )

        # Extract choices from the response
        choices = response.choices
        if choices:
            response_message = choices[0].message

            # Check if the function call exists
            function_call = getattr(response_message, 'function_call', None)
            if function_call:
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)

                if function_name in ['get_stock_price', 'calculate_RSI', 'calculate_MACD', 'plot_stock_price', 'predict_stock_price']:
                    if function_name == 'predict_stock_price':
                        args_dict = {
                            'ticker': function_args.get('ticker'),
                            'days_ahead': function_args.get('days_ahead')
                        }
                    else:
                        args_dict = {'ticker': function_args.get('ticker')}
                elif function_name in ['calculate_SMA', 'calculate_EMA']:
                    args_dict = {'ticker': function_args.get('ticker'), 'window': function_args.get('window')}

                function_to_call = available_functions[function_name]
                function_response = function_to_call(**args_dict)

                if function_name == 'plot_stock_price':
                    st.image('stock.png')
                else:
                    st.session_state['messages'].append(response_message)
                    st.session_state['messages'].append({
                        'role': 'function',
                        'name': function_name,
                        'content': function_response
                    })
                    # Generate final response
                    final_response = client.chat.completions.create(
                        messages=st.session_state['messages'],
                        model="llama3-70b-8192",
                        temperature=0.5,
                        max_tokens=1024,
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    final_message = final_response.choices[0].message
                    st.text(final_message.content)  # Access message content correctly
                    st.session_state['messages'].append({'role': 'assistant', 'content': final_message.content})
            else:
                # Handle case where function_call is None or not present
                st.text(response_message.content)
                st.session_state['messages'].append({'role': 'assistant', 'content': response_message.content})
        else:
            st.text('No choices found in response.')
    except Exception as e:
        st.text(f'Error: {str(e)}')
