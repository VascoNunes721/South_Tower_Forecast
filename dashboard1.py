import dash
from dash import html, dcc, callback_context
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn import  metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
import plotly.graph_objs as go
from sklearn import  linear_model
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import f_regression # score metric (f_regression)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

external_stylesheets = [
    'https://www.w3schools.com/w3css/4/w3.css',
    'https://fonts.googleapis.com/css?family=Lato',
    'https://fonts.googleapis.com/css?family=Montserrat',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
]

################################################################################################################

#Load data
df_1 = pd.read_csv('data_visualization.csv')
df_1['Date'] = pd.to_datetime (df_1['Date']) # create a new column 'data time' of datetime type

df_2 = pd.read_csv('features_analysis.csv')
df_2['Date'] = pd.to_datetime (df_2['Date']) # create a new column 'data time' of datetime type

df_3 = pd.read_csv('forecast_2019.csv')
df_3['Date'] = pd.to_datetime (df_3['Date']) # create a new column 'data time' of datetime type

available_metrics = ['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE']

#######################################################################################################################3


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    ################################################ MENU ################################################
    html.Div(className='w3-top', style={'background-color': '#009DE0'}, children=[
        html.Div(className='w3-bar w3-card w3-left-align w3-large', children=[
            html.Button(className='w3-bar-item w3-button w3-hide-medium w3-hide-large w3-right w3-padding-large w3-hover-white w3-large',
                        id='menu-button', children=[
                            html.I(className='fa fa-bars')
                        ]),
            html.A(className='w3-bar-item w3-button w3-padding-large w3-white', href='#', children='Home'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='#link1', children='Data Visualization'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='#link2', children='Exploratory Data Analysis'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='#link3', children='Feature Selection'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='#link4', children='Regression Model'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='#link5', children='Forecast')
        ]),
        html.Div(id='navDemo', className='w3-bar-block w3-white w3-hide w3-hide-large w3-hide-medium w3-large', children=[
            html.A(className='w3-bar-item w3-button w3-padding-large', href='#', children='Data Visualization'),
            html.A(className='w3-bar-item w3-button w3-padding-large', href='#', children='Exploratory Data Analysis'),
            html.A(className='w3-bar-item w3-button w3-padding-large', href='#', children='Feature Selection'),
            html.A(className='w3-bar-item w3-button w3-padding-large', href='#', children='Regression Model'),
            html.A(className='w3-bar-item w3-button w3-padding-large', href='#', children='Forecast')
        ])
    ]),
    ################################################ Header ################################################
    html.Header(className='w3-container w3-center', style={'padding': '128px 16px', 'background-color': '#009DE0'}, children=[
        html.H1(className='w3-margin w3-jumbo', children='SOUTH TOWER ENERGY FORECAST'),
        html.P(className='w3-xlarge', children='Energy Services Project by Vasco Nunes'),
    ]),
    ################################################ DATA VISUALIZATION ################################################
    html.Div(id='link1', className='w3-row-padding w3-padding-64 w3-container', children=[
        html.Div(className='w3-content', children=[
            html.Div(children=[ 
                    html.H1(children='Data Visualization'),
                    html.H5(className='w3-padding-32', style={'text-align': 'justify'}, children='Meteorological data plays a crucial role in forecasting energy consumption, providing valuable insights that enable more accurate predictions and efficient resource allocation. By integrating meteorological variables such as temperature, humidity, pressure, and radiance into energy consumption models, analysts can better understand the complex relationship between weather patterns and energy usage.'),
                    html.P(className='w3-text', style={'text-align': 'justify'}, children='Here, you can visualize energy consumption and meteorological data from 2017 to 2019 (up to March). Simply drag the slider to select the desired year and explore the variables you wish to inspect.'),
                    html.H4('Choose the year:'),
                    dcc.Slider(
                        min=2017,
                        max=2019,
                        marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(2017,2020)},
                        value=2017,
                        id='slider'),
                    html.Div([
                        dcc.Graph(id='yearly-data'),
                        ]),
            ])
        ])
    ]),    
    ################################################ DATA ANALYSIS ################################################
    html.Div(id='link2', className='w3-row-padding w3-light-grey w3-padding-64 w3-container', children=[
        html.Div(className='w3-content', children=[
            html.Div(className='w3-third', children=[
                html.H1(children='Exploratory Data Analysis'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='Here you can choose a year and a variable and display the data in three different formats.'),
                html.H5(className='w3-padding-32', children='Select options:'),
                html.Label('Choose the year:'),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[
                        {'label': '2017', 'value': 2017},
                        {'label': '2018', 'value': 2018},
                        {'label': '2019', 'value': 2019}
                    ],
                    value=2017
                ),
                html.Label('Choose a variable:'),
                dcc.Dropdown(
                    id='variable-dropdown',
                    options=[
                        {'label': 'Power (kW)', 'value': 'Power (kW)'},
                        {'label': 'Temperature (ºC)', 'value': 'Temperature (ºC)'},
                        {'label': 'Humidity (%)', 'value': 'Humidity (%)'},
                        {'label': 'Pressure (mbar)', 'value': 'Pressure (mbar)'},
                        {'label': 'Radiance (W/m^2)', 'value': 'Radiance (W/m^2)'}
                    ],
                    value='Power (kW)'
                ),
                html.Label('Choose a plot type:'),
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Linear Graph', 'value': 'line'},
                        {'label': 'Histogram', 'value': 'histogram'},
                        {'label': 'Boxplot', 'value': 'box'}
                    ],
                    value='line'
                )
            ]),
            html.Div(className='w3-twothird', children=[
                html.Div(id='plot-container')
            ])
        ])
    ]),
    ################################################ FEATURE SELECTION ################################################
    html.Div(id='link3', className='w3-row-padding w3-padding-64 w3-container', children=[
        html.Div(className='w3-content', children=[
            html.Div(className='w3-third', children=[
                html.H1(children='Features Selection'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='Here you can analyze how much each feature is related to the energy consumption by three different methods.'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='In addition to the original data, there are some new features related to previous power consumption and the date.'),
                html.H5(className='w3-padding-32', children='Select options:'),
                html.Label('Choose features:'),
                dcc.Dropdown(
                    id='features-dropdown',
                    options=[
                        {'label': 'Power-1', 'value': 'Power-1'},
                        {'label': 'Power deriv', 'value': 'Power deriv'},
                        {'label': 'Hour', 'value': 'Hour'},
                        {'label': 'Cos Hour', 'value': 'Cos Hour'},
                        {'label': 'Day/Night', 'value': 'Day/Night'},
                        {'label': 'Week Day', 'value': 'Week Day'},
                        {'label': 'Month', 'value': 'Month'},
                        {'label': 'Temperature (ºC)', 'value': 'Temperature (ºC)'},
                        {'label': 'Humidity (%)', 'value': 'Humidity (%)'},
                        {'label': 'Pressure (mbar)', 'value': 'Pressure (mbar)'},
                        {'label': 'Radiance (W/m^2)', 'value': 'Radiance (W/m^2)'}
                    ],
                    value=['Power-1','Power deriv','Cos Hour','Day/Night'],
                    multi=True  # Allow multiple selection
                ),
                html.Label('Choose the analysis method:'),
                dcc.Dropdown(
                    id='method-dropdown',
                    options=[
                        {'label': 'Filter Method', 'value': 'Filter Method'},
                        {'label': 'Wrapper Method', 'value': 'Wrapper Method'},
                        {'label': 'Ensemble Method', 'value': 'Ensemble Method'}
                    ],
                    value='Filter Method'
                )
            ]),
            html.Div(className='w3-twothird', children=[
                html.Div(id='features-plot')
            ])
        ])
    ]),
    ################################################  REGRESSION MODEL ################################################
    html.Div(id='link4', className='w3-row-padding w3-light-grey w3-padding-64 w3-container', children=[
        html.Div(className='w3-content', style={'display': 'flex', 'flexDirection': 'column'}, children=[
            html.Div(style={'flex': '1'}, children=[  # First half of the page
                html.H1(children='Regression Model Training'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='Here you will create an energy forecast model! Choose the features you think will work the best (from the analysis performed above) and choose one of the regression models.'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='The 2017 and 2018 data are randomly separated into training (75%) and test (25%) data. The graphs and table below show the performance of the model on the test data.'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='(Notice that this process can take a while.)'),
                html.H5(className='w3-padding-32', children='Select options:'),
                html.Div(className='w3-half', children=[
                    html.Label('Choose features:'),
                    dcc.Dropdown(
                        id='features-dropdown2',
                        options=[
                            {'label': 'Power-1', 'value': 'Power-1'},
                            {'label': 'Power deriv', 'value': 'Power deriv'},
                            {'label': 'Hour', 'value': 'Hour'},
                            {'label': 'Cos Hour', 'value': 'Cos Hour'},
                            {'label': 'Day/Night', 'value': 'Day/Night'},
                            {'label': 'Week Day', 'value': 'Week Day'},
                            {'label': 'Month', 'value': 'Month'},
                            {'label': 'Temperature (ºC)', 'value': 'Temperature (ºC)'},
                            {'label': 'Humidity (%)', 'value': 'Humidity (%)'},
                            {'label': 'Pressure (mbar)', 'value': 'Pressure (mbar)'},
                            {'label': 'Radiance (W/m^2)', 'value': 'Radiance (W/m^2)'}
                        ],
                        value=['Power-1','Power deriv','Cos Hour','Day/Night'],
                        multi=True  # Allow multiple selection
                    )
                ]),
                html.Div(className='w3-half', children=[
                    html.Label('Choose a regression model:'),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[
                            {'label': 'Linear Regression', 'value': 'Linear Regression'},
                            {'label': 'Decision Tree Regressor', 'value': 'Decision Tree Regressor'},
                            {'label': 'Random Forest', 'value': 'Random Forest'}
                        ],
                        value='Random Forest'
                    )
                ])
            ]),
            html.Div(style={'flex': '1'}, children=[  # Second half of the page
                html.Div(className='w3-half', children=[  
                    html.Div(id='model_training-plot')
                ]),
                html.Div(className='w3-half', children=[  
                    html.Div(id='model_error-plot')
                ]),
            ]),
            html.Div(className='w3-half', children=[
                html.Label('Choose metrics:'),
                dcc.Dropdown(
                    id='metrics-dropdown',
                    options=[{'label': metric, 'value': metric} for metric in available_metrics],
                    value=[],  # Default selected metrics
                    multi=True  # Allow multiple selection
                )
            ]),
            html.Div(children=[ 
                html.Div(id='table-plot')
            ])
        ])
    ]),
    ################################################  FORECAST  ################################################
    html.Div(id='link5', className='w3-row-padding w3-padding-64 w3-container', children=[
        html.Div(className='w3-content', style={'display': 'flex', 'flexDirection': 'column'}, children=[
            html.Div(style={'flex': '1'}, children=[  # First half of the page
                html.H1(children='Consumption Forecast'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='Here, your model is applied to new data (2019 data). The graphs and table below show the performance of the model.'),
            ]),
            html.Div(style={'flex': '1'}, children=[  # Second half of the page
                html.Div(className='w3-half', children=[  
                    html.Div(id='forecast-plot')
                ]),
                html.Div(className='w3-half', children=[  
                    html.Div(id='forecast_error-plot')
                ]),
            ]),
            html.Div(children=[ 
                html.Div(id='forecast_table-plot')
            ])
        ])
    ]),
    #######################################################
    html.Footer(className='w3-container w3-black w3-padding-64 w3-center w3-opacity', children=[
        html.P(children='Energy Services, IST, 3rd Term 2023/2024', style={'display': 'inline-block', 'verticalAlign': 'middle'})
    ])
])

###############################################################################################

########################################## Buttons
@app.callback(
    Output('navDemo', 'className'),
    [Input('menu-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_navbar(n_clicks):
    if n_clicks % 2 == 1:
        return 'w3-bar-block w3-white w3-show w3-large'
    else:
        return 'w3-bar-block w3-white w3-hide w3-large'

####################################### Data visualization
@app.callback(
    Output('yearly-data', 'figure'),
    [Input('slider', 'value')]
)
def update_figure(selected_year):
    # Filter the data based on the selected year
    filtered_df = df_1[df_1['Date'].dt.year == selected_year]
    
    # Create a new figure with the filtered data
    fig = px.line(filtered_df, x="Date", y=filtered_df.columns[1:6])

    fig.update_layout({
            "margin":{"l":0,"r":0,"t":50,"b":0}
        })
    
    return fig

######################################### Exploratory Data Analysis
@app.callback(
    Output('plot-container', 'children'),
    [Input('year-dropdown', 'value'),
     Input('variable-dropdown', 'value'),
     Input('plot-type-dropdown', 'value')]
)
def update_plot(selected_year, selected_variable, plot_type):
    # Filter the data based on the selected year
    filtered_df = df_1[df_1['Date'].dt.year == selected_year]
    
    # Generate the plot based on the selected variable and plot type
    if plot_type == 'line':
        fig = px.line(filtered_df, x='Date', y=selected_variable, title=f'{selected_variable} over time')
    elif plot_type == 'histogram':
        fig = px.histogram(filtered_df, x=selected_variable, title=f'{selected_variable} histogram')
    elif plot_type == 'box':
        fig = px.box(filtered_df, y=selected_variable, title=f'{selected_variable} boxplot')
    else:
        fig = None  # No plot
    
    # Update the plot layout to set the background color of the canvas
    if fig:
        fig.update_layout({
            'paper_bgcolor': '#f2f2f2',  # Set background color of the entire plot
            "margin":{"l":20,"r":0,"t":50,"b":0}
        })

    # Return the plot
    return dcc.Graph(figure=fig)

############################################# Feature Analysis
@app.callback(
    Output('features-plot', 'children'),
    [Input('features-dropdown', 'value'),
     Input('method-dropdown', 'value')]
)
def features_analysis_plot(selected_features, selected_method):
    
    Y = df_2.loc[:,'Power (kW)'].values
    X = df_2.loc[:, selected_features].values

    if (selected_method == 'Filter Method' and len(selected_features)!=0):
        features=SelectKBest(k=1,score_func=f_regression) 
        fit=features.fit(X,Y)
        df = pd.DataFrame({'features': selected_features, 'scores': fit.scores_})
        fig = px.bar(df, x='features', y='scores', title='Filter Method scores:')
    elif (selected_method == 'Wrapper Method' and len(selected_features)>1):
        model=LinearRegression() # LinearRegression Model as Estimator
        rfe=RFE(model,n_features_to_select=1)
        fit=rfe.fit(X,Y)
        df = pd.DataFrame({'features': selected_features, 'scores': fit.ranking_})
        fig = px.bar(df, x='features', y='scores', title='Wrapper method scores:')
    elif (selected_method == 'Ensemble Method' and len(selected_features)!=0):
        model = RandomForestRegressor()
        model.fit(X, Y)
        df = pd.DataFrame({'features': selected_features, 'scores': model.feature_importances_})
        fig = px.bar(df, x='features', y='scores', title='Ensemble method scores:')
    else:
        empty_df = pd.DataFrame({'features': [], 'scores': []})
        fig = px.bar(empty_df, x='features', y='scores', title='Choose more features.')
    
    fig.update_layout({
            "margin":{"l":20,"r":0,"t":50,"b":0}
        })

    # Return the plot
    return dcc.Graph(figure=fig)

############################################# Regression Model
@app.callback(
    [Output('model_training-plot', 'children'),
     Output('model_error-plot', 'children'),
     Output('forecast-plot', 'children'),
     Output('forecast_error-plot', 'children')],
    [Input('features-dropdown2', 'value'),
     Input('model-dropdown', 'value')]
)

def features_analysis_plot(selected_features, selected_model):

    Y = df_2.loc[:,'Power (kW)'].values
    X = df_2.loc[:, selected_features].values
    X_train, X_test, y_train, y_test = train_test_split(X,Y)
    Y2 = df_3.loc[:,'Power (kW)'].values
    X2 = df_3.loc[:, selected_features].values
    global MAE, MBE, MSE, RMSE, CVRMSE , NMBE, MAE_, MBE_, MSE_, RMSE_, CVRMSE_, NMBE_

    if (selected_model == 'Linear Regression' and len(selected_features)!=0):
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(X_train,y_train)
        # Make predictions using the testing set
        y_pred_LR = regr.predict(X_test)

        y_test_sampled = y_test[1:200]
        y_pred_LR_sampled = y_pred_LR[1:200]

        fig1 = px.line(y=y_test_sampled, title='Plot of the real data and the forecast')
        fig1.add_scatter(y=y_pred_LR_sampled, mode='lines', name='forecast')
        fig2 = px.scatter(x=y_test, y=y_pred_LR, title='Scatter Plot of the real data and the forecast')

        #Evaluate errors
        MAE=metrics.mean_absolute_error(y_test,y_pred_LR) 
        MBE=np.mean(y_test- y_pred_LR) #here we calculate MBE
        MSE=metrics.mean_squared_error(y_test,y_pred_LR)  
        RMSE= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
        CVRMSE=RMSE/np.mean(y_test)
        NMBE=MBE/np.mean(y_test)

        Model=regr
        y_pred = Model.predict(X2)
        fig3 = px.line(y=Y2, title='Plot of the real data and the forecast')
        fig3.add_scatter(y=y_pred, mode='lines', name='forecast')
        fig4 = px.scatter(x=Y2, y=y_pred, title='Scatter Plot of the real data and the forecast')
        #Evaluate errors
        MAE_=metrics.mean_absolute_error(Y2,y_pred) 
        MBE_=np.mean(Y2-y_pred) #here we calculate MBE
        MSE_=metrics.mean_squared_error(Y2,y_pred)  
        RMSE_= np.sqrt(metrics.mean_squared_error(Y2,y_pred))
        CVRMSE_=RMSE_/np.mean(Y2)
        NMBE_=MBE_/np.mean(Y2)
    
    elif (selected_model == 'Decision Tree Regressor' and len(selected_features)!=0):
        # Create Regression Decision Tree object
        DT_regr_model = DecisionTreeRegressor(min_samples_leaf=5)
        # Train the model using the training sets
        DT_regr_model.fit(X_train, y_train)
        # Make predictions using the testing set
        y_pred_DT = DT_regr_model.predict(X_test)

        y_test_sampled = y_test[1:200]
        y_pred_DT_sampled = y_pred_DT[1:200]

        fig1 = px.line(y=y_test_sampled, title='Plot of the real data and the forecast')
        fig1.add_scatter(y=y_pred_DT_sampled, mode='lines', name='forecast')
        fig2 = px.scatter(x=y_test, y=y_pred_DT, title='Scatter Plot of the real data and the forecast')

        MAE=metrics.mean_absolute_error(y_test,y_pred_DT) 
        MBE=np.mean(y_test-y_pred_DT) #here we calculate MBE
        MSE=metrics.mean_squared_error(y_test,y_pred_DT)  
        RMSE= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
        CVRMSE=RMSE/np.mean(y_test)
        NMBE=MBE/np.mean(y_test)

        Model=DT_regr_model
        y_pred = Model.predict(X2)
        fig3 = px.line(y=Y2, title='Plot of the real data and the forecast')
        fig3.add_scatter(y=y_pred, mode='lines', name='forecast')
        fig4 = px.scatter(x=Y2, y=y_pred, title='Scatter Plot of the real data and the forecast')
        #Evaluate errors
        MAE_=metrics.mean_absolute_error(Y2,y_pred) 
        MBE_=np.mean(Y2-y_pred) #here we calculate MBE
        MSE_=metrics.mean_squared_error(Y2,y_pred)  
        RMSE_= np.sqrt(metrics.mean_squared_error(Y2,y_pred))
        CVRMSE_=RMSE_/np.mean(Y2)
        NMBE_=MBE_/np.mean(Y2)
            
    elif (selected_model == 'Random Forest' and len(selected_features)!=0):
        parameters = {'bootstrap': True,
                    'min_samples_leaf': 3,
                    'n_estimators': 200, 
                    'min_samples_split': 15,
                    'max_features': 'sqrt',
                    'max_depth': 20,
                    'max_leaf_nodes': None}
        RF_model = RandomForestRegressor(**parameters)
        RF_model.fit(X_train, y_train)
        y_pred_RF = RF_model.predict(X_test)

        y_test_sampled = y_test[1:200]
        y_pred_RF_sampled = y_pred_RF[1:200]

        fig1 = px.line(y=y_test_sampled, title='Plot of the real data and the forecast')
        fig1.add_scatter(y=y_pred_RF_sampled, mode='lines', name='forecast')
        fig2 = px.scatter(x=y_test, y=y_pred_RF, title='Scatter Plot of the real data and the forecast')

        MAE=metrics.mean_absolute_error(y_test,y_pred_RF) 
        MBE=np.mean(y_test-y_pred_RF) #here we calculate MBE
        MSE=metrics.mean_squared_error(y_test,y_pred_RF)  
        RMSE= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
        CVRMSE=RMSE/np.mean(y_test)
        NMBE=MBE/np.mean(y_test)

        Model=RF_model
        y_pred = Model.predict(X2)
        fig3 = px.line(y=Y2, title='Plot of the real data and the forecast')
        fig3.add_scatter(y=y_pred, mode='lines', name='forecast')
        fig4 = px.scatter(x=Y2, y=y_pred, title='Scatter Plot of the real data and the forecast')
        #Evaluate errors
        MAE_=metrics.mean_absolute_error(Y2,y_pred) 
        MBE_=np.mean(Y2-y_pred) #here we calculate MBE
        MSE_=metrics.mean_squared_error(Y2,y_pred)  
        RMSE_= np.sqrt(metrics.mean_squared_error(Y2,y_pred))
        CVRMSE_=RMSE_/np.mean(Y2)
        NMBE_=MBE_/np.mean(Y2)        

    else:
        empty_df = pd.DataFrame({'features': [], 'scores': []})
        fig1 = px.bar(empty_df, x='features', y='scores', title='Choose more features.')
        fig2 = px.bar(empty_df, x='features', y='scores', title='Choose more features.')

        #Evaluate errors
        MAE=0
        MBE=0
        MSE=0
        RMSE=0
        CVRMSE=0
        NMBE=0

        fig3 = px.bar(empty_df, x='features', y='scores', title='Choose more features.')
        fig4 = px.bar(empty_df, x='features', y='scores', title='Choose more features.')
        #Evaluate errors
        MAE_=0
        MBE_=0
        MSE_=0  
        RMSE_=0
        CVRMSE_=0
        NMBE_=0
    
    # Update the plot layout to set the background color of the canvas
    if fig1:
        fig1.update_layout({
            'paper_bgcolor': '#f2f2f2',  # Set background color of the canvas
            "margin":{"l":0,"r":0,"t":75,"b":0}
        })
    if fig2:
        fig2.update_layout({
            'paper_bgcolor': '#f2f2f2',  # Set background color of the canvas
            "margin":{"l":0,"r":0,"t":75,"b":0}
        })
    if fig3:
        fig3.update_layout({
            "margin":{"l":0,"r":0,"t":75,"b":0}
        })
    if fig4:
        fig4.update_layout({
            "margin":{"l":0,"r":0,"t":75,"b":0}
        })

    # Return the plot
    return dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)

@app.callback(
    [Output('table-plot', 'children'),
     Output('forecast_table-plot', 'children')],
    [Input('metrics-dropdown', 'value')]
)

def tables_plot(selected_metrics):
        metrics_values1 = []
        metrics_values2 = []
        #Evaluate errors
        for metric in selected_metrics:
            if 'MAE' == metric:
                metrics_values1.append(MAE)
                metrics_values2.append(MAE_)
            elif 'MBE' == metric:
                metrics_values1.append(MBE)
                metrics_values2.append(MBE_)
            elif 'MSE' == metric:
                metrics_values1.append(MSE)
                metrics_values2.append(MSE_)
            elif 'RMSE' == metric:
                metrics_values1.append(RMSE)
                metrics_values2.append(RMSE_)
            elif 'cvRMSE' == metric:
                metrics_values1.append(CVRMSE)
                metrics_values2.append(CVRMSE_)
            elif 'NMBE' == metric:
                metrics_values1.append(NMBE)
                metrics_values2.append(NMBE_)

        # Calculate error metrics
        error_metrics_df = pd.DataFrame({
            'Metric': selected_metrics,#['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE']
            'Value': metrics_values1#[MAE, MBE, MSE, RMSE, cvRMSE, NMBE]
        })
        # Create a table figure using graph_objects
        table_figure = go.Figure(data=[go.Table(
            header=dict(values=list(error_metrics_df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[error_metrics_df.Metric, error_metrics_df.Value],
                    fill_color='lavender',
                    align='left'))
        ])

        # Set the background color of the table container
        table_figure.update_layout({
            'paper_bgcolor':'#f2f2f2',  # Set the background color to #f2f2f2
            "margin":{"l":0,"r":0,"t":40,"b":0}
        })

        # Calculate error metrics
        error_metrics_df2 = pd.DataFrame({
            'Metric': selected_metrics,#['MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE'],
            'Value': metrics_values2#[MAE_, MBE_, MSE_, RMSE_, cvRMSE_, NMBE_]
        })
        # Create a table figure using graph_objects
        table_figure2 = go.Figure(data=[go.Table(
            header=dict(values=list(error_metrics_df2.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[error_metrics_df2.Metric, error_metrics_df2.Value],
                    fill_color='lavender',
                    align='left'))
        ])
        # Set the background color of the table container
        table_figure2.update_layout({
            "margin":{"l":0,"r":0,"t":40,"b":0}
        })


        return dcc.Graph(figure=table_figure,style={'height': '200px'}), dcc.Graph(figure=table_figure2,style={'height': '200px'})

if __name__ == '__main__':
    app.run_server()
