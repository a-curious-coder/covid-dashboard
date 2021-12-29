from IPython.display import display
import plotly.io as pio
from datetime import date
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
import dash
import datetime
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from countryinfo import CountryInfo
import os
import glob
from os.path import exists
pd.options.mode.chained_assignment = None  # default='warn'

verbose = True
generate_new_files = False

app = dash.Dash(__name__)

server = app.server
# ---------------------------------------------------------------
app.layout = html.Div([
    html.Div([
        html.Label(['Select a Country']),
        dcc.Dropdown(
            id='my_dropdown',
            options=[
                {'label': 'Global', 'value': 'Global'},
                {'label': 'Afghanistan', 'value': 'Afghanistan'},
                {'label': 'Albania', 'value': 'Albania'},
                {'label': 'Algeria', 'value': 'Algeria'},
                {'label': 'Andorra', 'value': 'Andorra'},
                {'label': 'Angola', 'value': 'Angola'},
                {'label': 'Antigua and Barbuda', 'value': 'Antigua and Barbuda'},
                {'label': 'Argentina', 'value': 'Argentina'},
                {'label': 'Armenia', 'value': 'Armenia'},
                {'label': 'Australia', 'value': 'Australia'},
                {'label': 'Austria', 'value': 'Austria'},
                {'label': 'Azerbaijan', 'value': 'Azerbaijan'},
                {'label': 'Bahamas', 'value': 'Bahamas'},
                {'label': 'Bahrain', 'value': 'Bahrain'},
                {'label': 'Bangladesh', 'value': 'Bangladesh'},
                {'label': 'Barbados', 'value': 'Barbados'},
                {'label': 'Belarus', 'value': 'Belarus'},
                {'label': 'Belgium', 'value': 'Belgium'},
                {'label': 'Belize', 'value': 'Belize'},
                {'label': 'Benin', 'value': 'Benin'},
                {'label': 'Bhutan', 'value': 'Bhutan'},
                {'label': 'Bolivia', 'value': 'Bolivia'},
                {'label': 'Bosnia and Herzegovina',
                    'value': 'Bosnia and Herzegovina'},
                {'label': 'Botswana', 'value': 'Botswana'},
                {'label': 'Brazil', 'value': 'Brazil'},
                {'label': 'Brunei', 'value': 'Brunei'},
                {'label': 'Bulgaria', 'value': 'Bulgaria'},
                {'label': 'Burkina Faso', 'value': 'Burkina Faso'},
                {'label': 'Burma', 'value': 'Burma'},
                {'label': 'Burundi', 'value': 'Burundi'},
                {'label': 'Cabo Verde', 'value': 'Cabo Verde'},
                {'label': 'Cambodia', 'value': 'Cambodia'},
                {'label': 'Cameroon', 'value': 'Cameroon'},
                {'label': 'Canada', 'value': 'Canada'},
                {'label': 'Central African Republic',
                    'value': 'Central African Republic'},
                {'label': 'Chad', 'value': 'Chad'},
                {'label': 'Chile', 'value': 'Chile'},
                {'label': 'China', 'value': 'China'},
                {'label': 'Colombia', 'value': 'Colombia'},
                {'label': 'Comoros', 'value': 'Comoros'},
                {'label': 'Congo (Brazzaville)',
                 'value': 'Congo (Brazzaville)'},
                {'label': 'Congo (Kinshasa)', 'value': 'Congo (Kinshasa)'},
                {'label': 'Costa Rica', 'value': 'Costa Rica'},
                {'label': 'Cote d\'Ivoire', 'value': 'Cote d\'Ivoire'},
                {'label': 'Croatia', 'value': 'Croatia'},
                {'label': 'Cuba', 'value': 'Cuba'},
                {'label': 'Cyprus', 'value': 'Cyprus'},
                {'label': 'Czechia', 'value': 'Czechia'},
                {'label': 'Denmark', 'value': 'Denmark'},
                {'label': 'Diamond Princess', 'value': 'Diamond Princess'},
                {'label': 'Djibouti', 'value': 'Djibouti'},
                {'label': 'Dominica', 'value': 'Dominica'},
                {'label': 'Dominican Republic', 'value': 'Dominican Republic'},
                {'label': 'Ecuador', 'value': 'Ecuador'},
                {'label': 'Egypt', 'value': 'Egypt'},
                {'label': 'El Salvador', 'value': 'El Salvador'},
                {'label': 'Equatorial Guinea', 'value': 'Equatorial Guinea'},
                {'label': 'Eritrea', 'value': 'Eritrea'},
                {'label': 'Estonia', 'value': 'Estonia'},
                {'label': 'Eswatini', 'value': 'Eswatini'},
                {'label': 'Ethiopia', 'value': 'Ethiopia'},
                {'label': 'Fiji', 'value': 'Fiji'},
                {'label': 'Finland', 'value': 'Finland'},
                {'label': 'France', 'value': 'France'},
                {'label': 'Gabon', 'value': 'Gabon'},
                {'label': 'Gambia', 'value': 'Gambia'},
                {'label': 'Georgia', 'value': 'Georgia'},
                {'label': 'Germany', 'value': 'Germany'},
                {'label': 'Ghana', 'value': 'Ghana'},
                {'label': 'Greece', 'value': 'Greece'},
                {'label': 'Grenada', 'value': 'Grenada'},
                {'label': 'Guatemala', 'value': 'Guatemala'},
                {'label': 'Guinea', 'value': 'Guinea'},
                {'label': 'Guinea-Bissau', 'value': 'Guinea-Bissau'},
                {'label': 'Guyana', 'value': 'Guyana'},
                {'label': 'Haiti', 'value': 'Haiti'},
                {'label': 'Holy See', 'value': 'Holy See'},
                {'label': 'Honduras', 'value': 'Honduras'},
                {'label': 'Hungary', 'value': 'Hungary'},
                {'label': 'Iceland', 'value': 'Iceland'},
                {'label': 'India', 'value': 'India'},
                {'label': 'Indonesia', 'value': 'Indonesia'},
                {'label': 'Iran', 'value': 'Iran'},
                {'label': 'Iraq', 'value': 'Iraq'},
                {'label': 'Ireland', 'value': 'Ireland'},
                {'label': 'Israel', 'value': 'Israel'},
                {'label': 'Italy', 'value': 'Italy'},
                {'label': 'Jamaica', 'value': 'Jamaica'},
                {'label': 'Japan', 'value': 'Japan'},
                {'label': 'Jordan', 'value': 'Jordan'},
                {'label': 'Kazakhstan', 'value': 'Kazakhstan'},
                {'label': 'Kenya', 'value': 'Kenya'},
                {'label': 'Kiribati', 'value': 'Kiribati'},
                {'label': 'Korea, South', 'value': 'Korea, South'},
                {'label': 'Kosovo', 'value': 'Kosovo'},
                {'label': 'Kuwait', 'value': 'Kuwait'},
                {'label': 'Kyrgyzstan', 'value': 'Kyrgyzstan'},
                {'label': 'Laos', 'value': 'Laos'},
                {'label': 'Latvia', 'value': 'Latvia'},
                {'label': 'Lebanon', 'value': 'Lebanon'},
                {'label': 'Lesotho', 'value': 'Lesotho'},
                {'label': 'Liberia', 'value': 'Liberia'},
                {'label': 'Libya', 'value': 'Libya'},
                {'label': 'Liechtenstein', 'value': 'Liechtenstein'},
                {'label': 'Lithuania', 'value': 'Lithuania'},
                {'label': 'Luxembourg', 'value': 'Luxembourg'},
                {'label': 'MS Zaandam', 'value': 'MS Zaandam'},
                {'label': 'Madagascar', 'value': 'Madagascar'},
                {'label': 'Malawi', 'value': 'Malawi'},
                {'label': 'Malaysia', 'value': 'Malaysia'},
                {'label': 'Maldives', 'value': 'Maldives'},
                {'label': 'Mali', 'value': 'Mali'},
                {'label': 'Malta', 'value': 'Malta'},
                {'label': 'Marshall Islands', 'value': 'Marshall Islands'},
                {'label': 'Mauritania', 'value': 'Mauritania'},
                {'label': 'Mauritius', 'value': 'Mauritius'},
                {'label': 'Mexico', 'value': 'Mexico'},
                {'label': 'Micronesia', 'value': 'Micronesia'},
                {'label': 'Moldova', 'value': 'Moldova'},
                {'label': 'Monaco', 'value': 'Monaco'},
                {'label': 'Mongolia', 'value': 'Mongolia'},
                {'label': 'Montenegro', 'value': 'Montenegro'},
                {'label': 'Morocco', 'value': 'Morocco'},
                {'label': 'Mozambique', 'value': 'Mozambique'},
                {'label': 'Namibia', 'value': 'Namibia'},
                {'label': 'Nepal', 'value': 'Nepal'},
                {'label': 'Netherlands', 'value': 'Netherlands'},
                {'label': 'New Zealand', 'value': 'New Zealand'},
                {'label': 'Nicaragua', 'value': 'Nicaragua'},
                {'label': 'Niger', 'value': 'Niger'},
                {'label': 'Nigeria', 'value': 'Nigeria'},
                {'label': 'North Macedonia', 'value': 'North Macedonia'},
                {'label': 'Norway', 'value': 'Norway'},
                {'label': 'Oman', 'value': 'Oman'},
                {'label': 'Pakistan', 'value': 'Pakistan'},
                {'label': 'Palau', 'value': 'Palau'},
                {'label': 'Panama', 'value': 'Panama'},
                {'label': 'Papua New Guinea', 'value': 'Papua New Guinea'},
                {'label': 'Paraguay', 'value': 'Paraguay'},
                {'label': 'Peru', 'value': 'Peru'},
                {'label': 'Philippines', 'value': 'Philippines'},
                {'label': 'Poland', 'value': 'Poland'},
                {'label': 'Portugal', 'value': 'Portugal'},
                {'label': 'Qatar', 'value': 'Qatar'},
                {'label': 'Romania', 'value': 'Romania'},
                {'label': 'Russia', 'value': 'Russia'},
                {'label': 'Rwanda', 'value': 'Rwanda'},
                {'label': 'Saint Kitts and Nevis',
                    'value': 'Saint Kitts and Nevis'},
                {'label': 'Saint Lucia', 'value': 'Saint Lucia'},
                {'label': 'Saint Vincent and the Grenadines',
                    'value': 'Saint Vincent and the Grenadines'},
                {'label': 'Samoa', 'value': 'Samoa'},
                {'label': 'San Marino', 'value': 'San Marino'},
                {'label': 'Sao Tome and Principe',
                    'value': 'Sao Tome and Principe'},
                {'label': 'Saudi Arabia', 'value': 'Saudi Arabia'},
                {'label': 'Senegal', 'value': 'Senegal'},
                {'label': 'Serbia', 'value': 'Serbia'},
                {'label': 'Seychelles', 'value': 'Seychelles'},
                {'label': 'Sierra Leone', 'value': 'Sierra Leone'},
                {'label': 'Singapore', 'value': 'Singapore'},
                {'label': 'Slovakia', 'value': 'Slovakia'},
                {'label': 'Slovenia', 'value': 'Slovenia'},
                {'label': 'Solomon Islands', 'value': 'Solomon Islands'},
                {'label': 'Somalia', 'value': 'Somalia'},
                {'label': 'South Africa', 'value': 'South Africa'},
                {'label': 'South Sudan', 'value': 'South Sudan'},
                {'label': 'Spain', 'value': 'Spain'},
                {'label': 'Sri Lanka', 'value': 'Sri Lanka'},
                {'label': 'Sudan', 'value': 'Sudan'},
                {'label': 'Summer Olympics 2020', 'value': 'Summer Olympics 2020'},
                {'label': 'Suriname', 'value': 'Suriname'},
                {'label': 'Sweden', 'value': 'Sweden'},
                {'label': 'Switzerland', 'value': 'Switzerland'},
                {'label': 'Syria', 'value': 'Syria'},
                {'label': 'Taiwan*', 'value': 'Taiwan*'},
                {'label': 'Tajikistan', 'value': 'Tajikistan'},
                {'label': 'Tanzania', 'value': 'Tanzania'},
                {'label': 'Thailand', 'value': 'Thailand'},
                {'label': 'Timor-Leste', 'value': 'Timor-Leste'},
                {'label': 'Togo', 'value': 'Togo'},
                {'label': 'Tonga', 'value': 'Tonga'},
                {'label': 'Trinidad and Tobago', 'value': 'Trinidad and Tobago'},
                {'label': 'Tunisia', 'value': 'Tunisia'},
                {'label': 'Turkey', 'value': 'Turkey'},
                {'label': 'US', 'value': 'US'},
                {'label': 'Uganda', 'value': 'Uganda'},
                {'label': 'Ukraine', 'value': 'Ukraine'},
                {'label': 'United Arab Emirates', 'value': 'United Arab Emirates'},
                {'label': 'United Kingdom', 'value': 'United Kingdom'},
                {'label': 'Uruguay', 'value': 'Uruguay'},
                {'label': 'Uzbekistan', 'value': 'Uzbekistan'},
                {'label': 'Vanuatu', 'value': 'Vanuatu'},
                {'label': 'Venezuela', 'value': 'Venezuela'},
                {'label': 'Vietnam', 'value': 'Vietnam'},
                {'label': 'West Bank and Gaza', 'value': 'West Bank and Gaza'},
                {'label': 'Yemen', 'value': 'Yemen'},
                {'label': 'Zambia', 'value': 'Zambia'},
                {'label': 'Zimbabwe', 'value': 'Zimbabwe'}
            ],
            value='Global',
            multi=False,
            clearable=False,
            style={"width": "50%"}
        ),
    ]),

    html.Div([
        dcc.Graph(id='dashboard')
    ])
])

# ---------------------------------------------------------------


@app.callback(
    Output(component_id='dashboard', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)
def update_graph(country):
    # Calculate today / yesterday
    today = date.today()
    # Today's data's latest entry will have yesterday's date
    yesterday = today - datetime.timedelta(days=1)
    # Format dates
    today = today.strftime("%d_%m_%Y")
    yesterday = yesterday.strftime("%d_%m_%Y")

    dprint("[*]\tUpdating graph")

    # Read in cleaned esri data (All Time Statistics)
    df = pd.read_csv("cleaned_esri_data.csv")

    labels = ["Confirmed", "Deaths", "Recovered"]
    stats = get_country_stats(df, country)
    df_copy = pd.DataFrame()
    dprint("[*]\tPlotting Data")
    if country != "Global":
        df_copy = df[df["Country_Region"] == country]
        del df_copy[df_copy.columns[0]]
    else:
        values = get_country_stats(df, country)
        df_copy = pd.DataFrame({"Confirmed": [values[0]], "Deaths": [
                               values[2]], "Recovered": [values[1]]})

    # Transpose table for single column table
    df_copy = df_copy.T
    # Rename column from index 0 to country name
    df_copy = df_copy.rename(columns={df_copy.columns[0]: country})
    # Replace '0' values with nan values so they're not needlessly labelled in plot
    df_copy = df_copy.replace(0, np.nan)

    fig = make_subplots(
        rows=5, cols=6,
        specs=[
            [{"type": "pie", "rowspan": 2, "colspan": 2}, None, {"type": "indicator"}, {
                "type": "indicator"}, {"type": "indicator"}, None],
            [None, None, {"type": "bar", "colspan": 3}, None, None, None],
            [None, None, {"type": "bar", "colspan": 3}, None, None, None],
            [None, None, {"type": "bar", "colspan": 3}, None, None, None],
            [None, None, {"type": "bar", "colspan": 3}, None, None, None]
        ]
    )

    fig = country_cases_deaths_pie_chart(fig, country, labels, df_copy[country])
    
    # TODO: Make this malleable for any 'country'
    # Read data
    
    df = pd.DataFrame()
    if country == "Global":
        print("Global")
        df = pd.read_csv(f"covid_data_{today}.csv")
        df = transform_owid_covid_data(df)
    else:
        print(country)
        if exists(f"covid_data_clean_{country}_{today}.csv") == False:
            print("Generating data")
            generate_covid_data2(country)
        df = pd.read_csv(f"covid_data_clean_{country}_{today}.csv")
        print(df['Country'].head(5))
    # If country's data doesn't exist, keep global
    if df.shape[0] == 0:
        print(f"{country} not available")
        df = pd.read_csv(f"covid_data_{today}.csv")
        df = transform_owid_covid_data(df)
        country = f"(Country {country} not available) : Global"
    # Get country population
    pop = 0
    if "Global" not in country:
        print(country)
        pop = CountryInfo(country).population()
    else:
        # World pop
        pop = 7300000000
    
    dprint("[*]\tPlotting indicators")
    # Add 'vaccinated once' indicator
    fig = vaccinated_once_indicator(fig, df, pop, row = 1, col = 3)
    # Add 'vaccinated_fully" indicator
    fig = vaccinated_fully_indicator(fig, df, pop, row = 1, col = 4)
    # Add 'boosted' indicator
    fig = boosted_indicator(fig, df, pop, row = 1, col = 5)
    # # Add Bar chart showing top ten countries with highest fully vaccinations
    fig = top_ten_vaccinated(fig, today, yesterday, row= 2, col = 3)

    fig.layout.height = 700
    # fig.update_layout(
    #     margin={
    #         't': 50,
    #         'b': 0,
    #         'r': 0,
    #         'l': 0,
    #         'pad': 200,
    #     }
    # )
    print("[*]\tPlotted")
    return (fig)


# Non-infected, infected, deaths
def country_cases_deaths_pie_chart(fig, country, labels, values):
    values = values.tolist()
    if country != "Global":
        pop = CountryInfo(country).population()
    else:
        pop = 7300000000
        
    labels.append("Non-Infected")
    values.append(pop - values[0] - values[1])


    # Pie chart
    fig.add_trace(
        go.Pie(title=str(country), labels=labels, values=values, hole=.5),
        row=1, col=1)
    # Replaces percentages with actual values
    # fig.update_traces(textinfo='value+label',
    #                   textfont_size=12,
    #                   marker=dict(line=dict(color='#000000', width=1)))
    return fig

def vaccinated_once_indicator(fig, df, population, row, col):
    vaccination_count = int(df["Vaccinated_Once"].head(1))
    fig.add_trace(
        go.Indicator(
            mode="number",
            number=dict(suffix="%"),
            value=vaccination_count/population*100,
            title=f"Vaccinated Once"
        ),
        row=row, col=col
    )
    return fig

def vaccinated_fully_indicator(fig, df, population, row, col):
    vaccination_count = int(df["Vaccinated_Full"].head(1))
    fig.add_trace(
        go.Indicator(
            mode="number",
            number=dict(suffix="%"),
            value=vaccination_count/population*100,
            title=f"Fully Vaccinated"
        ),
        row=row, col=col
    )
    return fig

def boosted_indicator(fig, df, population, row, col):
    vaccination_count = int(df["Vaccinated_Booster"].head(1))
    fig.add_trace(
        go.Indicator(
            mode="number",
            number=dict(suffix="%"),
            value=vaccination_count/population*100,
            title="Booster"
        ),
        row=row, col=col
    )
    return fig

def top_ten_vaccinated(fig, date, yesterday, row, col):
    # TODO: refine this with percentages of country population that's vaccinated
    # Calculate the countries with top 10 leading fully vaccinated
    data = get_covid_data(date)
    
    df_countries = []
    # For each country
    for country in data['location'].unique():
        # Update NaN/0 values to latest vacc value
        fixed = update_null_vacc_values(data[data['location'] == country], [
                                        'people_fully_vaccinated'])
        # Append update null vacc values
        df_countries.append(fixed)

    data = pd.concat(df_countries)
    # Filter data to only include latest entries
    data = data[data['date'] == str(yesterday)]

    data = data.sort_values(by="people_fully_vaccinated",
                            ascending=False).head(10)
    new_data = pd.DataFrame(
        {"Country": data['location'], "Vacc": data['people_fully_vaccinated']})

    countries = data['location'].tolist()
    cases = data['people_fully_vaccinated'].tolist()
    cases = [int(x) for x in cases]

    fig.add_trace(go.Bar(x=countries,
                         y=cases,
                         name='Country'), row=2, col=3)
    print("bar")
    return fig
    
def get_esri_covid_data():
    """
    Gets the latest covid 19 data
    :return:
    """
    if exists("cleaned_esri_data.csv") == False:
        # Download CSV data from ESRI
        raw = requests.get(
            "https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer"
            "/1/query?where=1%3D1&outFields=*&outSR=4326&f=json")
        raw_json = raw.json()
        # Store data into dataframe
        df = pd.DataFrame(raw_json["features"])
        # Transform data
        df = transform_data(df)
        dprint("[*]\tGetting COVID-19 Data")
        df = clean_esri_data(df)
        df = get_sum_statistics(df)
        df.to_csv("cleaned_esri_data.csv", index=False)
    df = pd.read_csv("cleaned_esri_data.csv")
    return df


def clean_esri_data(data):
    cleaned_data = data.dropna(subset=["Last_Update"])
    cleaned_data["Province_State"].fillna(value="", inplace=True)

    cleaned_data["Last_Update"] = cleaned_data["Last_Update"]/1000
    cleaned_data["Last_Update"] = cleaned_data["Last_Update"].apply(
        convert_time)
    dprint("[*]\tCleaning Data")
    return cleaned_data


def generate_covid_data():
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    # If there's no file with today's statistics, download it
    if exists(f"covid_data_{d1}.csv") == False:
        download_covid_data()

    if generate_new_files or exists(f"covid_data_clean_UK_{d1}.csv") == False:
        # Get today's covid data
        df = get_covid_data(d1)
        # Clean data
        # df = clean_owid_data(df)
        # Filter dataframe to only include country stats
        df = df[df["location"] == "United Kingdom"]
        # Save uncleaned data
        df.to_csv(f"covid_data_unclean_UK_{d1}.csv", index=False)
        # Refine covid dataframe
        df = transform_owid_covid_data(df)
        # Save data to csv
        df.to_csv(f"covid_data_clean_UK_{d1}.csv", index=False)

    df = pd.read_csv(f"covid_data_clean_UK_{d1}.csv")
    return df

def generate_covid_data2(country):
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    # If there's no file with today's statistics, download it
    if exists(f"covid_data_{d1}.csv") == False:
        download_covid_data()

    if generate_new_files or exists(f"covid_data_clean_{country}_{d1}.csv") == False:
        # Get today's covid data
        df = get_covid_data(d1)
        # Clean data
        # df = clean_owid_data(df)
        # Filter dataframe to only include country stats
        df = df[df["location"] == country]
        # Save uncleaned data
        df.to_csv(f"covid_data_unclean_{country}_{d1}.csv", index=False)
        # Refine covid dataframe
        df = transform_owid_covid_data(df)
        # Save data to csv
        df.to_csv(f"covid_data_clean_{country}_{d1}.csv", index=False)
    df = pd.read_csv(f"covid_data_clean_{country}_{d1}.csv")
    
    return df


def download_covid_data():
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    dprint(f"[*]\Downloading OWID COVID-19 Data for {d1}")
    csv_url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    req = requests.get(csv_url)
    url_content = req.content
    csv_file = open(f"covid_data_{d1}.csv", 'wb')
    csv_file.write(url_content)
    csv_file.close()
    df = pd.read_csv(f"covid_data_{d1}.csv")
    df = clean_owid_data(df)
    df.to_csv(f"covid_data_{d1}.csv", index = False)


def format_and_sort_by_date(df, date):
    try:
        df[date] = pd.to_datetime(df[date], format='%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates are already formatted correctly")
    
    df.sort_values(by=date, inplace=True, ascending=False)
    df[date] = df[date].dt.strftime('%d_%m_%Y')
    return df


def get_covid_data(date):
    # Should be cleaned already
    # Read file to dataframe
    df = pd.read_csv(f"covid_data_{date}.csv")
    return df


def clean_owid_data(data):
    # Format and sort by date
    data = format_and_sort_by_date(data, "date")
    # Replace null values with value '0'
    data = data.fillna(0)
    # Remove continents and anything that isn't a country
    continents = ["Asia", "Africa", "North America", "South America",
                  "Antarctica", "Europe", "Australia", "European Union"]
    for continent in continents:
        data = data[data['location'] != continent]
    data = data[data['location'] != "World"]
    data = data[data['location'] != "Upper middle income"]
    data = data[data['location'] != "High income"]
    data = data[data['location'] != "Lower middle income"]

    # TODO: Sort out vaccinated count
    # updated_vacc = []
    # # 'Vaccinated' columns accumulate value each day - change 0s to max value
    # vaccinated_columns = ["people_vaccinated", "people_fully_vaccinated", "total_boosters"]
    # for country in data['location'].unique():
    #     for column in vaccinated_columns:
    #         fixed = update_null_vacc_values(country, data, column)
    #     updated_vacc.append(fixed)
    #     pd.concat(updated_vacc)
    return data


def transform_owid_covid_data(data):
    dprint("[*]\tTransforming OWID Data")
    dprint(data.shape)
    # Initialising dataframe with data we're interested in
    clean_data = pd.DataFrame(
        {"Country": data["location"],
         "Confirmed": data["new_cases"],
         "Deaths": data["new_deaths"],
         "Hosp_patients": data["hosp_patients"],
         "Vaccinations": data["new_vaccinations"],
         "Vaccinated_Once": data["people_vaccinated"],
         "Vaccinated_Full": data["people_fully_vaccinated"],
         "Vaccinated_Booster": data["total_boosters"],
         "Date": data["date"]})
    return clean_data


def update_null_vacc_values(clean_data, vaccinated_columns):
    # Caters for single country only
    # For each vaccination column
    for column in vaccinated_columns:
        # Store column as list
        vaccinated_data = clean_data[column].tolist()
        # Only check for '0' values in first 100 rows
        entries = 100
        for i, value in enumerate(vaccinated_data):
            # When we've hit the limit of entries
            if i == entries:
                continue
            if value == 0:
                vaccinated_data[i] = max(vaccinated_data)
        # Apply new cleaned list to dataframe
        clean_data[column] = vaccinated_data
    return clean_data


def clean_directory():
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")

    dprint("[*]\tChecking for old files")
    for filename in glob.glob("*.csv"):
        if "covid_data" in filename:
            if d1 not in filename:
                os.remove(filename)


def get_sum_statistics(data):
    """
    Extracts and summates the confirmed, recovered and deseased case values for each country
    :param data:
    :return:
    """
    df_total = data.groupby("Country_Region", as_index=False).agg(
        {
            "Confirmed": "sum",
            "Deaths": "sum",
            "Recovered": "sum"
        }
    )
    return df_total


def transform_data(data):
    """
Filters the data to return the information we want
:param data:
:return:
"""
    dprint("[*]\tTransforming Data")
    data_list = data["attributes"].tolist()
    df_final = pd.DataFrame(data_list)
    df_final.set_index("OBJECTID")
    df_final = df_final[
        ["Country_Region", "Province_State", "Lat", "Long_", "Confirmed", "Deaths", "Recovered", "Last_Update"]]
    return df_final


def convert_time(t):
    """
Converts time in milliseconds to datetime format
:param t:
:return:
"""
    t = int(t)
    return datetime.datetime.fromtimestamp(t)


def dprint(text):
    """
Prints if verbose mode is enabled
:param text:
:return:
"""
    if verbose:
        print(text)


def get_country_stats(data, country):
    if country != "Global":
        data = data[data["Country_Region"] == country]

    confirmed = data["Confirmed"].sum()
    recovered = data["Recovered"].sum()
    deaths = data["Deaths"].sum()
    stats = [confirmed, recovered, deaths]
    return stats


def main():
    dprint("-------------------------------")
    clean_directory()
    generate_covid_data()
    get_esri_covid_data()
    dprint("[*]\tData Ready")


if __name__ == "__main__":
    main()
    app.run_server(debug=True)
