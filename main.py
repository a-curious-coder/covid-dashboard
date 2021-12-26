import requests
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
import plotly.io as pio
from IPython.display import display

verbose = True


def get_covid_data():
	"""
    Gets the latest covid 19 data
    :return:
    """
	dprint("Getting COVID-19 Data")
	raw = requests.get(
		"https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Coronavirus_2019_nCoV_Cases/FeatureServer"
		"/1/query?where=1%3D1&outFields=*&outSR=4326&f=json")
	raw_json = raw.json()
	df = pd.DataFrame(raw_json["features"])
	df = transform_data(df)
	return df


def get_sum_statistics(data):
	"""
	Extracts and summates the confirmed, recovered and deseased case values for each country
	:param data:
	:return:
	"""
	df_total = data.groupby("Country_Region", as_index=False).agg(
		{
			"Confirmed" : "sum",
			"Deaths" : "sum",
			"Recovered" : "sum"
		}
	)
	return df_total


def transform_data(data):
	"""
    Filters the data to return the information we want
    :param data:
    :return:
    """
	dprint("Transforming Data")
	data_list = data["attributes"].tolist()
	df_final = pd.DataFrame(data_list)
	df_final.set_index("OBJECTID")
	df_final = df_final[
		["Country_Region", "Province_State", "Lat", "Long_", "Confirmed", "Deaths", "Recovered", "Last_Update"]]
	return df_final


def clean_data(data):
    dprint("Cleaning Data")
    cleaned_data = data.dropna(subset=["Last_Update"])
    cleaned_data["Province_State"].fillna(value="", inplace=True)

    cleaned_data["Last_Update"]= cleaned_data["Last_Update"]/1000
    cleaned_data["Last_Update"] = cleaned_data["Last_Update"].apply(convert_time)
    
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


def plot_global_case_statistics(df_final):
	dprint("Plotting data")
	total_confirmed = df_final["Confirmed"].sum()
	total_recovered = df_final["Recovered"].sum()
	total_deaths = df_final["Deaths"].sum()
	labels = ["Confirmed", "Recovered", "Deaths"]
	values = [total_confirmed, total_recovered, total_deaths]
	fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, )])
	fig.update_layout(
		plot_bgcolor='rgb(255,255,255)',
		paper_bgcolor ='rgb(211,211,211)'
		)
	pio.write_html(fig, file='index.html', auto_open=True)
	# fig.show()
	
	
def main():
	df = get_covid_data()
	df = clean_data(df)
	df = get_sum_statistics(df)
	plot_global_case_statistics(df)
	
	# display(df.head(5))
	print(f"Done {df.shape}")


if __name__ == "__main__":
	main()
