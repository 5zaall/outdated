################################################################################
# Filename          : flask_app.py
# 
# Description       : This file is responsible for running the Flask framework .
#
# Created by        : J Faizal Gulam Dastagir on 29 November 2023
# Last Modified by  : -
#
################################################################################


################################################################################
# IMPORTS
################################################################################

# ------------------------------------------------------------------------------
# Standard Python Libraries
# ------------------------------------------------------------------------------

import sys
import json
import mariadb
import yfinance as yf
import pandas as pd
import random
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import finplot as fplt

from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from io import BytesIO
import base64


from waitress       import serve
from hashlib        import sha256
from configobj      import ConfigObj
from datetime       import timedelta
from datetime       import date


from flask          import request, session, flash
from flask          import Flask, redirect, url_for, render_template
from flask          import Flask, request, render_template, jsonify

from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Package for additional features
from talib import abstract as ta
from talib import RSI
from talib import MACD
import finnhub # Extract economical news 


################################################################################
# CLASSES & METHODS, FUNCTIONS, ROUTES
################################################################################

# ==============================================================================
# FlaskApp Class (Start) 
# ==============================================================================

class FlaskApp:
    """ This class is responsible for running the Flask framework.
    """
    
    # ==========================================================================
    # Constructor
    # ==========================================================================

    #####
    def __init__(self):
        """ Initialized FlaskApp Class.
        """

        # ----------------------------------------------------------------------
        # Class Instances
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Public Variables
        # ----------------------------------------------------------------------

        ''' None '''

        # ----------------------------------------------------------------------
        # Private Variables
        # ----------------------------------------------------------------------

        # Flask
        self.__flask_app                        = None
        self.__flask_app_host                   = None
        self.__flask_app_port                   = None
        #self.__flask_app_mode                   = "DEVELOPMENT" 
        self.__flask_app_mode                   = "PRODUCTION"     

        # Config
        self.__flask_app_config                 = ConfigObj("config.txt")

        # Encryption 
        self.__flask_app_salt                   = None

        # Database
        self.__flask_app_connection             = None
        self.__flask_app_cursor                 = None

        # Temporary watchlist 
        self.watchlist                          = set()

        # ----------------------------------------------------------------------
        # Internal Method Calls
        # ----------------------------------------------------------------------
      
        self.read_config()   

    # ==========================================================================
    # Public Methods
    # ==========================================================================

    def read_config(self):
        """ This method reads the operator console configuration from the config file.

        Returns:
            [type]: none
        """

        # // Need to configure database settings and api settings later on 
        try:
            # Read Database Settings

            # change according to your database settings 
            config = {
                'host': 'StockSage.mysql.pythonanywhere-services.com',
                'port': 3306,
                'user': 'StockSage',
                'password': 'fypfyp1234',
                'database': 'StockSage$default'
            }

            self.__flask_app_connection = mariadb.connect(**config)

            # Read encription Settings
            #self.__flask_app_salt = self.__flask_app_config['encription']['salt']

            # Read API Settings
            #self.__flask_app_host= self.__flask_app_config['api']['host']
            #self.__flask_app_port = int(self.__flask_app_config['api']['port'])

        except mariadb.Error as e:
            print(f"Error connecting to MariaDB Platform: {e}")
            sys.exit(1)

    #####
    def start_flask(self):

        #// for database 
        # Get Cursor
        self.__flask_app_cursor = self.__flask_app_connection.cursor(dictionary = True)

        self.__flask_app = Flask(__name__)
        self.__flask_app.secret_key = "1234"

        # ----------------------------------------------------------------------
        # Flask Route (START)
        # ----------------------------------------------------------------------

        # landing page
        @self.__flask_app.route("/", methods=["POST", "GET"])
        def landingPage():
            return self.__landing_page()
        
        # about page
        @self.__flask_app.route("/about", methods=["POST", "GET"])
        def aboutPage():
            return self.__about_page()
        
        # work page
        @self.__flask_app.route("/work", methods=["POST", "GET"])
        def workPage():
            return self.__work_page()
        
        # team page
        @self.__flask_app.route("/team", methods=["POST", "GET"])
        def teamPage():
            return self.__team_page()
        
        # contact page
        @self.__flask_app.route("/contact", methods=["POST", "GET"])
        def contactPage():
            return self.__contact_page()
        
        # login page
        @self.__flask_app.route("/login", methods=["POST", "GET"])
        def login():
            return self.__page_login()
            
        # register page
        @self.__flask_app.route("/register", methods=["POST", "GET"])
        def registerUser():
            return self.__page_register()
        
        # create admin page
        @self.__flask_app.route("/createAdmin", methods=["POST", "GET"])
        def createAdmin():
            return self.__page_createAdmin()
        
        # home page    
        @self.__flask_app.route("/home", methods=["POST", "GET"])
        def home():
            return self.__page_home()
        
        # Route to search stock    
        @self.__flask_app.route("/search", methods=["POST", "GET"])
        def search_stock():
            return self.__search_stock()
        
        # Route to add a symbol to the watchlist
        @self.__flask_app.route('/add_to_watchlist/<symbol>')
        def add_to_watchlist(symbol):
            self.watchlist.add(symbol)
            db_watchlist = ', '.join(self.watchlist)
            username = session.get("name")

            self.__flask_app_cursor.execute(
                            "UPDATE user SET watchlist = ? WHERE username = ?",
                            (db_watchlist, username)
                        )
            self.__flask_app_connection.commit()
            
            return redirect(url_for('watchList'))
  
        # Route to remove a symbol from the watchlist
        @self.__flask_app.route('/remove_from_watchlist/<symbol>')
        def remove_from_watchlist(symbol):
            self.watchlist.remove(symbol)
            db_watchlist = ', '.join(self.watchlist)
            username = session.get("name")

            self.__flask_app_cursor.execute(
                            "UPDATE user SET watchlist = ? WHERE username = ?",
                            (db_watchlist, username)
                        )
            self.__flask_app_connection.commit()
            return redirect(url_for('watchList'))
        
        # watch list page
        @self.__flask_app.route("/watchList", methods=["POST", "GET"])
        def watchList():
            return self.__page_watchList()
            
        # profile page
        @self.__flask_app.route("/profile", methods=["POST", "GET"])
        def profile():
            return self.__page_profile()
        
        @self.__flask_app.route('/stock_detail/<symbol>')
        def stock_detail(symbol):
           return self.__page_stock_details(symbol)
        

        # Route for the prediction page
        @self.__flask_app.route('/prediction')
        def prediction():
            return self.__page_prediction()
        
        # Route for the prediction page
        @self.__flask_app.route('/predict', methods=['POST'])
        def predict():
            return self.__predict()
        
        # update profile route
        @self.__flask_app.route("/update_profile", methods=["POST"])
        def update_profile():
            return self.__page_update_profile()
            
        @self.__flask_app.route("/quote")
        def display_quote():
            # get a stock ticker symbol from the query string
            # default to AAPL
            symbol = request.args.get('symbol', default="AAPL")

            # pull the stock quote
            quote = yf.Ticker(symbol)

            #return the object via the HTTP Response
            return jsonify(quote.info)

        # API route for pulling the stock history
        @self.__flask_app.route("/history")
        def display_history():
            #get the query string parameters
            symbol = request.args.get('symbol', default="AAPL")
            period = request.args.get('period', default="1y")
            interval = request.args.get('interval', default="1mo")

            #pull the quote
            quote = yf.Ticker(symbol)	
            #use the quote to pull the historical data from Yahoo finance
            hist = quote.history(period=period, interval=interval)
            #convert the historical data to JSON
            data = hist.to_json()
            #return the JSON in the HTTP response
            return data

        # manage users page
        @self.__flask_app.route("/manageUsers", methods=["POST", "GET"])
        def manageUsers():
            return self.__page_manageUsers()
        
        # View enquiries page
        @self.__flask_app.route("/viewEnquiries", methods=["POST", "GET"])
        def viewEnquiries():
            return self.__page_viewEnquiries()
        
        # View edit page
        @self.__flask_app.route("/edit", methods=["POST", "GET"])
        def edit():
            return self.__page_edit()
        
        # Search route
        @self.__flask_app.route("/search", methods=["POST", "GET"])
        def search():
            return self.__page_search()

        # logout        
        @self.__flask_app.route("/logout")
        def logout():
            return self.__page_logout()
        
        # ----------------------------------------------------------------------
        # Flask Route (END)
        # ----------------------------------------------------------------------

        if self.__flask_app_mode.upper() == "DEVELOPMENT":
            # Werkzeug WSGI for Development Server #
            self.__flask_app.run(host=self.__flask_app_host, port=self.__flask_app_port, debug= True, use_reloader= True)
            self.__flask_app.run(host=self.__flask_app_host, port=self.__flask_app_port)
        
        elif self.__flask_app_mode.upper() == "PRODUCTION":
            # Waitress WSGI for Production Server #
            serve(self.__flask_app, host=self.__flask_app_host, port=self.__flask_app_port)

        else:
            pass  # Do nothing.
    

    # ==========================================================================
    # Private Methods
    # ==========================================================================
    
    # --------------------------------------------------------------------------
    # Flask Route Contents (START)
    # --------------------------------------------------------------------------


    ##### 
    def evaluate_model(self, X, y, model):
        tscv = TimeSeriesSplit(n_splits=10)
        mse_scores = []
        mae_scores = []
        r2_scores = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mse_scores.append(mse)
            mae_scores.append(mae)
            r2_scores.append(r2)

        return np.mean(mse_scores), np.mean(mae_scores), np.mean(r2_scores)

    def news_sentiment(self, end_date, company_code):

        api_key = "cn0ah7pr01qkcvkfucv0cn0ah7pr01qkcvkfucvg"; 

        finnhub_client = finnhub.Client(api_key=api_key)

        # Get all the news of that day for the company
        data = finnhub_client.company_news(company_code, _from =end_date, to=end_date)

        return data
    
    def __landing_page(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        return render_template("landingPage.html")
    
    def __about_page(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        return render_template("about.html")
    
    def __work_page(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        return render_template("work.html")
    
    def __team_page(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        return render_template("team.html")
    
    def __contact_page(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        if request.method == "POST":

            input_name = request.form["name"]
            input_email = request.form["email"]
            input_message = request.form["message"]
            print(input_name)
            print(input_email)
            print(input_message)
     

            self.__flask_app_cursor.execute(
                "INSERT INTO enquiries (name,email,message) VALUES (?, ?, ?)",
                    (input_name, input_email, input_message),
            )

            self.__flask_app_connection.commit()

            flash("Enquiry Sent Successfully!")
            return redirect(url_for("contactPage"))
                
        else:
            return render_template("contact.html")



    #####
    def __page_login(self):
        """ This method reads user input for login details
            and allows user to log in after checking credentials.
        """

        if request.method == "POST":

           
            session["name"] = request.form.get("name")
            session["password"] = request.form.get("Password")
                
            user = request.form["name"]
            input_password = request.form["Password"]
                
            valid_user = False
                
            #//use after setting up encryption 
            # salt = self.__flask_app_salt
            # input_ = salt + input_password
            # password_string_to_db = sha256(input_.encode('utf-8')).hexdigest()


            if user == "" or input_password == "":
                flash("User Name or Password cannot be empty!", "info")
                return redirect(url_for("login"))

            self.__flask_app_cursor.execute(
                "SELECT username FROM user"
            )

            for name in self.__flask_app_cursor.fetchall():
                if name["username"] == user :                   
                    valid_user = True

            if valid_user == True:
                self.__flask_app_cursor.execute(
                    "SELECT password FROM user WHERE username = ? ",
                    (user,)
                )

                for password in self.__flask_app_cursor.fetchall():
                    if input_password == password["password"]:

                        #//this code block is used to distinguish users with diffeent permissions
                        #//can be used later on in the project 
                        self.__flask_app_cursor.execute(
                            "SELECT permission FROM user WHERE username = ?",
                            (user,)
                        )
                        for permission in self.__flask_app_cursor.fetchall():
                            session["permission"] = permission["permission"]

                        #for the time being, permission will always be admin
                        #session["permission"] = "Admin"
                        #directs user to main page after successful login 
                                
                        # List of 10 popular stock symbols
                        stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'DIS', 'NVDA', 'BA', 'NFLX', 'INTC']

                        # Create an empty DataFrame to store the data
                        all_data = pd.DataFrame()

                        today = date.today()
                        yesterday = today - timedelta(days = 1)
                        yesterday2 = today - timedelta(days = 3)
                        print(today)

                        # Fetch historical data for each stock and concatenate it to the DataFrame
                        for symbol in stock_symbols:
                            stock_data = yf.download(symbol, start=yesterday, end=today, progress=False)
                            stock_data['Symbol'] = symbol  # Add a column for the stock symbol
                            all_data = pd.concat([all_data, stock_data])

                        # Reset index to make 'Date' a regular column
                        all_data.reset_index(inplace=True)

                        # Drop the 'Date' column
                        all_data.drop(columns='Date', inplace=True)

                        # Reorder columns with "Symbol" as the first column
                        columns_order = ['Symbol'] + [col for col in all_data.columns if col != 'Symbol']
                            
                        # Reset the index to start from 1
                        all_data.index = all_data.index + 1
                        all_data = all_data[columns_order]

                        # Convert the DataFrame to HTML
                        table_html = all_data.to_html(classes='table table-striped')
                        return render_template("home.html", data=all_data.to_dict('records'))
                        
                    else:
                        flash(
                            "User Name or Password incorrect! Login Unsuccesful!"
                            , "info"
                        )
                        return redirect(url_for("login"))
            else:
                flash("User Name does not exist!", "info")
                return redirect(url_for("login"))
        else:
            return render_template("login.html")


    def __page_register(self):


        if request.method == "POST":

            input_fullname = request.form["name"]
            input_username = request.form["username"]
            input_email = request.form["email"]
            input_password = request.form["Password"]
            confirm_password = request.form["ConfirmPassword"]
            risk_status = request.form["Risk-Status"]
            age_range = request.form["Age"]
            user_exists = False

            self.__flask_app_cursor.execute("SELECT username FROM user")

            for username in self.__flask_app_cursor:

                if username['username']  == input_username:
                    user_exists = True
                   

            if user_exists == False:

                if input_password == confirm_password:
                    self.__flask_app_cursor.execute(
                        "INSERT INTO user (username,password,email,fullname,permission,riskstatus,agerange) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (input_username, input_password, input_email, input_fullname, "Member", risk_status, age_range),
                    )

                    self.__flask_app_connection.commit()

                    flash("Registered Successfully!")
                    return redirect(url_for("login"))

                else:
                    flash("Password and Confirm password does not match!")
                    return redirect(url_for("registerUser"))
            else:
                flash("User already Exists!")
                return redirect(url_for("registerUser"))
                
        else:
            return render_template("register.html")

    def __page_createAdmin(self):

        if not session.get("name"):
            return redirect("/")
        elif(session["permission"] != "Admin"):
            return redirect("/home")

        if request.method == "POST":

            input_fullname = request.form["name"]
            input_username = request.form["username"]
            input_email = request.form["email"]
            input_password = request.form["Password"]
            confirm_password = request.form["ConfirmPassword"]
            risk_status = request.form["Risk-Status"]
            age_range = request.form["Age"]
            user_exists = False

            self.__flask_app_cursor.execute("SELECT username FROM user")

            for username in self.__flask_app_cursor:

                if username['username']  == input_username:
                    user_exists = True
                   

            if user_exists == False:

                if input_password == confirm_password:
                    self.__flask_app_cursor.execute(
                        "INSERT INTO user (username,password,email,fullname,permission,riskstatus,agerange) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (input_username, input_password, input_email, input_fullname, "Admin", risk_status, age_range),
                    )

                    self.__flask_app_connection.commit()

                    flash("Admin Created Successfully!")
                    return redirect(url_for("createAdmin"))

                else:
                    flash("Password and Confirm password does not match!")
                    return redirect(url_for("createAdmin"))
            else:
                flash("User already Exists!")
                return redirect(url_for("createAdmin"))
                
        else:
            return render_template("createAdmin.html")
        
    def __page_profile(self):
        """ This method is for profile page
        """

        username = session.get("name")
        self.__flask_app_cursor.execute(
                "SELECT * FROM user WHERE username = ?", (username,)
            )
        
        user = self.__flask_app_cursor.fetchone()
        if not session.get("name"):

            return redirect("/")

        else:
            
            return render_template("profile.html", user = user)

        
    def __page_home(self):

        """ This method is for home page
        """
        if not session.get("name"):
            return redirect("/")
            
        else:
            # List of 10 popular stock symbols
            stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'DIS', 'NVDA', 'BA', 'NFLX', 'INTC']

            # Create an empty DataFrame to store the data
            all_data = pd.DataFrame()

            today = date.today()
            yesterday = today - timedelta(days = 1)
            yesterday2 = today - timedelta(days = 3)

            # Fetch historical data for each stock and concatenate it to the DataFrame
            for symbol in stock_symbols:
                stock_data = yf.download(symbol, start=yesterday, end=today, progress=False)

                stock_data['Symbol'] = symbol  # Add a column for the stock symbol
                all_data = pd.concat([all_data, stock_data])

            # Reset index to make 'Date' a regular column
            all_data.reset_index(inplace=True)

            # Drop the 'Date' column
            all_data.drop(columns='Date', inplace=True)

            # Reorder columns with "Symbol" as the first column
            columns_order = ['Symbol'] + [col for col in all_data.columns if col != 'Symbol']
            
            # Reset the index to start from 1
            all_data.index = all_data.index + 1
            all_data = all_data[columns_order]

            return render_template("home.html", data=all_data.to_dict('records'))
        
    def __search_stock(self):

        """ This method is for home page
        """
        if not session.get("name"):
            return redirect("/")
            
        else:
            # List of 10 popular stock symbols
            stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'DIS', 'NVDA', 'BA', 'NFLX', 'INTC']

            # Create an empty DataFrame to store the data
            all_data = pd.DataFrame()

            today = date.today()
            yesterday = today - timedelta(days = 1)
            yesterday2 = today - timedelta(days = 3)

            # Fetch historical data for each stock and concatenate it to the DataFrame
            for symbol in stock_symbols:
                stock_data = yf.download(symbol, start=yesterday, end=today, progress=False)

                stock_data['Symbol'] = symbol  # Add a column for the stock symbol
                all_data = pd.concat([all_data, stock_data])

            # Reset index to make 'Date' a regular column
            all_data.reset_index(inplace=True)

            # Drop the 'Date' column
            all_data.drop(columns='Date', inplace=True)

            # Reorder columns with "Symbol" as the first column
            columns_order = ['Symbol'] + [col for col in all_data.columns if col != 'Symbol']
            
            # Reset the index to start from 1
            all_data.index = all_data.index + 1
            all_data = all_data[columns_order]


            # Create an empty DataFrame to store the data
            search_details = pd.DataFrame()
            search_symbol = request.form['search_symbol']
            print("THIS IS SEARCHED SYMBOL")
            print(search_symbol)
            search_data = yf.download(search_symbol, start=yesterday, end=today, progress=False)
            search_data['Symbol'] = search_symbol  # Add a column for the stock symbol
            search_details = pd.concat([search_details, search_data])

            return render_template("home.html", data=all_data.to_dict('records'), searchDetails=search_details.to_dict('records'))

    def __page_watchList(self):

        """ This method is for watch list page
        """
        if not session.get("name"):
            return redirect("/")
            
        else:
             # Create an empty DataFrame to store the data
            all_data = pd.DataFrame()

            today = date.today()
            yesterday = today - timedelta(days = 1)
            yesterday2 = today - timedelta(days = 3)

            username = session.get("name")
            print(username)
            self.__flask_app_cursor.execute(
                    "SELECT watchlist FROM user WHERE username = ? ",
                    (username,)
                )
     
            db_watchlist = (self.__flask_app_cursor.fetchall()[0])['watchlist']   
                   
            if(db_watchlist != ''):
                self.watchlist = set(str(db_watchlist).split(", "))
            else:
                self.watchlist = set()
            

            if(len(self.watchlist) != 0):
            # Fetch historical data for each stock and concatenate it to the DataFrame
                for symbol in self.watchlist:
                    stock_data = yf.download(symbol, start=yesterday, end=today, progress=False)

                    stock_data['Symbol'] = symbol  # Add a column for the stock symbol
                    all_data = pd.concat([all_data, stock_data])

                # Reset index to make 'Date' a regular column
                all_data.reset_index(inplace=True)

                # Drop the 'Date' column
                all_data.drop(columns='Date', inplace=True)

                # Reorder columns with "Symbol" as the first column
                columns_order = ['Symbol'] + [col for col in all_data.columns if col != 'Symbol']
                
                # Reset the index to start from 1
                all_data.index = all_data.index + 1
                all_data = all_data[columns_order]

                return render_template("watchlist.html", watchlist = self.watchlist, data=all_data.to_dict('records'))
            else:
                return render_template("watchlist.html", watchlist = self.watchlist, data={})

    def __page_stock_details(self, symbol):

        if not session.get("name"):
            return redirect("/")
            
        else:    
            # Fetch basic information for the specified symbol
            stock_info = yf.Ticker(symbol).info

            # Fetch news for the specified symbol
            stock_news = yf.Ticker(symbol).news

             # Fetch historical stock prices for the symbol
            historical_data = yf.Ticker(symbol).history(period='1y')

            # Create an interactive candlestick chart using Plotly
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.7, 0.3])

            # Candlestick chart
            candlestick = go.Candlestick(x=historical_data.index,
                                        open=historical_data['Open'],
                                        high=historical_data['High'],
                                        low=historical_data['Low'],
                                        close=historical_data['Close'],
                                        name='Candlesticks')

            fig.add_trace(candlestick, row=1, col=1)

            # Volume chart
            volume = go.Bar(x=historical_data.index, y=historical_data['Volume'], name='Volume', marker=dict(color='blue'))

            fig.add_trace(volume, row=2, col=1)

            # Set layout
            fig.update_layout(
                            xaxis_rangeslider_visible=False,
                            height=600)

            # Convert the plot to HTML
            candlestick_chart = fig.to_html(full_html=False)

            

            # Render the symbol detail page with information and news
            return render_template('stock_detail.html', symbol=symbol, stock_info=stock_info, stock_news=stock_news, candlestick_chart=candlestick_chart)

    def __page_prediction(self):

        if not session.get("name"):
            return redirect("/")
            
        else:
            # Render the symbol detail page with information and news
            return render_template('prediction.html')

    def __predict(self):

        if not session.get("name"):
            return redirect("/")
            
        else:
            # Load the pre-trained LSTM model
            model = load_model('model.h5')

            ticker_symbol = request.form['ticker_symbol']
            end_date = request.form['end_date']

            # Fetch historical data
            start_date = "2010-01-01"
            data = yf.download(ticker_symbol, start=start_date, end=end_date)
            df = pd.DataFrame(data)

            # Feature engineering and preprocessing
            # Relative Strength Index (RSI): price movement over a given period.
            df['RSI'] = RSI(df['Close'], timeperiod=14)

            # Moving Average Convergence Divergence (MACD): Identifies trend strength and potential turning points.
            macd, signal, hist = MACD(df['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['MACD'] = macd
            df['Signal'] = signal
            df['MACD_Hist'] = hist

            close_prices = df["Adj Close"]
            high_prices = df["High"]
            low_prices = df["Low"]

            true_range = pd.Series(
                [max(hi - lo, abs(hi - close_prev), abs(lo - close_prev))
                for hi, lo, close_prev in zip(high_prices, low_prices, close_prices.shift(1))]
            )

            # Common window size, which can balance.
            window = 14
            atr = true_range.rolling(window=window).mean()

            atr_df = pd.DataFrame({'ATR': atr.values}, index=df.index)

            # Merge the original DataFrame with the new ATR DataFrame
            df = pd.merge(df, atr_df, left_index=True, right_index=True)

            # Re-order the data frame
            new_order = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR", "Adj Close"]
            df = df[new_order]

            # Drop null values
            df.dropna(inplace=True)

            # Set Target Variable
            output_var = pd.DataFrame(df["Adj Close"])

            # Selecting the Features
            features = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR"]

            # Scaling
            scaler = MinMaxScaler()
            feature_transform = scaler.fit_transform(df[features])
            feature_transform = pd.DataFrame(data=feature_transform, columns=features, index=df.index)

            # Selecting relevant features and scaling
            features = ["Open", "High", "Low", "Volume", "RSI", 'MACD', 'Signal', 'MACD_Hist', "ATR"]
            scaler = MinMaxScaler()
            feature_transform = scaler.fit_transform(df[features])

            # Reshape data for LSTM input
            X = np.array(feature_transform)
            X = X.reshape(X.shape[0], 1, X.shape[1])

            # Make predictions
            predictions = model.predict(X)

            # Plot out

            # Create an interactive plot with Plotly
            fig = make_subplots(rows=1, cols=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Past Stock Prices'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[end_date], y=predictions[-1], mode='markers', marker=dict(color='red', size=8), name='Predicted Price'), row=1, col=1)
            fig.update_layout(title='Past Stock Prices and Predicted Price', xaxis_title='Date', yaxis_title='Stock Price', showlegend=True)

           
            plot_html = fig.to_html(full_html=False)

            # Extract the last predicted price
            predicted_price = predictions[-1][0]

            # Display the prediction and accuracy
            mse_avg, mae_avg, r2_avg = self.evaluate_model(X, df['Adj Close'].values, model)
            accuracy = r2_avg * 100

            # Display the news 

            #headlines = [data[i]['headline'] for i in range(len(news_sentiment(end_date, ticker_symbol)))]


            # Render to the template 
            return render_template('prediction.html',
                                prediction=f'Predicted price for {ticker_symbol} on {end_date}: {predicted_price:.2f}',
                                accuracy=f'Accuracy: {accuracy:.2f}',
                                mse_avg = mse_avg,
                                mae_avg = mae_avg,
                                r2_avg = r2_avg, 
                                plot_html = plot_html,
                                # headlines = headlines
                                )


    def __page_update_profile(self):

        if not session.get("name"):
            return redirect("/")
            
        else: 
            if request.method == "POST":
                    userid = request.form.get("userid")
                    username = request.form.get("username")
                    password = request.form.get("Password")
                    email = request.form.get("email")
                    fullname = request.form.get("fullname")
                    risk_status = request.form["Risk-Status"]
                    age_range = request.form["Age"]

                    if request.form['action'] == "update":
                        self.__flask_app_cursor.execute(
                            "UPDATE user SET username = ?, password = ?, email = ?, fullname = ?, riskstatus = ?, agerange = ? WHERE userid = ?",
                            (username, password, email, fullname, risk_status, age_range, userid)
                        )
                        self.__flask_app_connection.commit()
                    
                        return redirect("/login")
                    
                    elif request.form['action'] == "delete":
                        self.__flask_app_cursor.execute(
                            "DELETE FROM user WHERE userid = ?", (userid,)
                        )
                        self.__flask_app_connection.commit()

                        return redirect("/login")   


    def __page_manageUsers(self):
        """ This method is for view users page
        """
        username = session.get("name")

        if(session["permission"] == "Admin"):
            self.__flask_app_cursor.execute(
                    "SELECT * FROM user "
                )
        else:
            self.__flask_app_cursor.execute(
                    "SELECT * FROM user WHERE permission = 'Member'"
                )
        users = self.__flask_app_cursor.fetchall()

        if not session.get("name"):
            return redirect("/")
        elif(session["permission"] != "Admin"):
            return redirect("/home")    
        else:
            
            return render_template("manageUsers.html", users=users)
        
    def __page_viewEnquiries(self):
        """ This method is for view enquiries page
        """
        username = session.get("name")
        self.__flask_app_cursor.execute(
                "SELECT * FROM enquiries"
            )
        
        enquiries = self.__flask_app_cursor.fetchall()

        if not session.get("name"):
            return redirect("/")
        elif(session["permission"] != "Admin"):
            return redirect("/home")    
        else:
            
            return render_template("enquiries.html", enquiries = enquiries)

    def __page_edit(self):
        """ This method is for edit page
        """

        userid = request.args.get("userid")
        self.__flask_app_cursor.execute(
                "SELECT * FROM user WHERE userid = ?", (userid,)
            )
        
        user = self.__flask_app_cursor.fetchone()

        if not session.get("name"):
            return redirect("/") 
        else:
            
            return render_template("profile.html", user=user)

    def __page_search(self):
        """ This method is for search
        """

        search_term = request.form.get("searchbar")
        search_pattern = "%"+ search_term +"%"
        self.__flask_app_cursor.execute(
                "SELECT * FROM user WHERE username LIKE ? AND permission = 'Member'", (search_pattern,)
            )
        
        users = self.__flask_app_cursor.fetchall()
        if not users:
            self.__flask_app_cursor.execute(
                "SELECT * FROM user WHERE permission = 'Member'"
            )
        
            users = self.__flask_app_cursor.fetchall()
        

        if not session.get("name"):
            return redirect("/")
            
        else:
            print(search_term)
            return render_template("viewUsers.html", users=users)    

    def __page_logout(self):
        """ This method logs the user out and redirects to the landing page
        """
        session["name"] = None
        return redirect("/")

    # --------------------------------------------------------------------------
    # Flask Route Contents (END)
    # --------------------------------------------------------------------------

# ==============================================================================
# FlaskApp Class (End) 
# ==============================================================================


# ==============================================================================
# Program Entry Point (Start) 
# ==============================================================================

if __name__ == "__main__":

    app = FlaskApp()
    app.start_flask()

# ==============================================================================
# Program Entry Point (End) 
# ==============================================================================



################################################################################
# END OF FILE
################################################################################