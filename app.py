from IPython.display import display
import plotly.io as pio
from datetime import date
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import requests
import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

verbose = True
df = pd.DataFrame()

app = dash.Dash(__name__)

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
            value='United Kingdom',
            multi=False,
            clearable=False,
            style={"width": "50%"}
        ),
    ]),

    html.Div([
        dcc.Graph(id='pie')
    ])
])

# ---------------------------------------------------------------


@app.callback(
    Output(component_id='pie', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)
def update_graph(my_dropdown_choice):
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    labels = ["Confirmed", "Recovered", "Deaths"]
    stats = get_country_stats(df, my_dropdown_choice)
    df_copy = pd.DataFrame()
    dprint("[*]\tPlotting Data")
    if my_dropdown_choice != "Global":
        df_copy = df[df["Country_Region"] == my_dropdown_choice]
        del df_copy[df_copy.columns[0]]
    else:
        values = get_country_stats(df, my_dropdown_choice)
        df_copy = pd.DataFrame({"Confirmed": [values[0]], "Recovered": [
                               values[1]], "Deaths": [values[2]]})

    df_copy = df_copy.T
    df_copy = df_copy.rename(columns={df_copy.columns[0]: my_dropdown_choice})

    piechart = px.pie(
        title=f"{my_dropdown_choice}",
        data_frame=df_copy,
        names=["Confirmed", "Deaths", "Recovered"],
        values=my_dropdown_choice,
        hole=.7
    )

    piechart.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        title_x=0.5,
        title_y=0.475
    )
    # pio.write_html(piechart, file='index.html', auto_open=True)
    return (piechart)


def get_covid_data():
    """
Gets the latest covid 19 data
:return:
"""
    raw = requests.get(
        "https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer"
        "/1/query?where=1%3D1&outFields=*&outSR=4326&f=json")
    raw_json = raw.json()
    df = pd.DataFrame(raw_json["features"])
    df = transform_data(df)
    dprint("[*]\tGetting COVID-19 Data")
    return df


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


def clean_data(data):
    cleaned_data = data.dropna(subset=["Last_Update"])
    cleaned_data["Province_State"].fillna(value="", inplace=True)

    cleaned_data["Last_Update"] = cleaned_data["Last_Update"]/1000
    cleaned_data["Last_Update"] = cleaned_data["Last_Update"].apply(
        convert_time)
    dprint("[*]\tCleaning Data")
    return cleaned_data


def convert_time(t):
    """
Converts time in milliseconds to datetime format
:param t:
:return:
"""
    t = int(t)
    return datetime.fromtimestamp(t)


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


def plot_global_case_statistics(df_final):
    dprint("Plotting data")
    total_confirmed = df_final["Confirmed"].sum()
    total_recovered = df_final["Recovered"].sum()
    total_deaths = df_final["Deaths"].sum()
    labels = ["Confirmed", "Recovered", "Deaths"]
    values = [total_confirmed, total_recovered, total_deaths]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, )])
    fig.update_layout(
        title="Plotted",
        plot_bgcolor='rgb(255,255,255)',
        paper_bgcolor='rgb(211,211,211)',
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["values", values],
                        label="Global",
                    ),
                    dict(
                        args=["values", get_country_stats(df_final, "United Kingdom")],
                        label="United Kingdom",
                    )
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )]
    )
    pio.write_html(fig, file='index.html', auto_open=True)
    # fig.show()


def main():
    global df
    df = get_covid_data()
    df = clean_data(df)
    df = get_sum_statistics(df)
    # plot_global_case_statistics(df)
    # for value in df["Country_Region"].unique():
    #     print("{\'label\': \'" + value + "\', \'value\': \'" + value + "\'},")
    print(f"[*]\tData Ready")


if __name__ == "__main__":
    main()
    app.run_server(debug=True)
