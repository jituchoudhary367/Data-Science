import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly.figure_factory import create_table
from wordcloud import WordCloud

# ✅ Set Plotly renderer 
pio.renderers.default = 'browser'
# pio.renderers.default = 'vscode'

# ✅ Load datasets
dataset1 = pd.read_csv("covid.csv")
dataset2 = pd.read_csv("covid_grouped.csv")
dataset3 = pd.read_csv("coviddeath.csv")


# ✅ Dataset metadata
print(dataset1.shape)
print(dataset1.size)
dataset1.info()

print(dataset2.shape)
print(dataset2.size)
dataset2.info()

# ✅ Clean dataset1
dataset1.drop(["NewCases", "NewDeaths", "NewRecovered"], axis=1, inplace=True)

# ✅ Table
table = create_table(dataset1.head(15))
pio.show(table)

# ✅ Bar charts
px.bar(dataset1.head(15), x="Country/Region", y="TotalCases", color="TotalCases", height=500, hover_data=['Country/Region', "Continent"]).show()
px.bar(dataset1.head(15), x='Country/Region', y='TotalCases', color='TotalDeaths', height=500, hover_data=['Country/Region', 'Continent']).show()
px.bar(dataset1.head(15), x='Country/Region', y='TotalCases', color='TotalTests', height=500, hover_data=['Country/Region', 'Continent']).show()
px.bar(dataset1.head(15), x='TotalTests', y='Country/Region', color='TotalTests', orientation='h', height=500, hover_data=['Country/Region', 'Continent']).show()
px.bar(dataset1.head(15), x='TotalTests', y='Continent', color='TotalTests', orientation='h', height=500, hover_data=['Country/Region', 'Continent']).show()

# ✅ Scatter plots
px.scatter(dataset1, x='Continent', y='TotalCases', hover_data=['Country/Region'], color='TotalCases', size='TotalCases', size_max=80).show()
px.scatter(dataset1.head(57), x='Continent', y='TotalCases', hover_data=['Country/Region'], color='TotalCases', size='TotalCases', size_max=80, log_y=True).show()
px.scatter(dataset1.head(54), x='Continent', y='TotalTests', hover_data=['Country/Region'], color='TotalTests', size='TotalTests', size_max=80).show()
px.scatter(dataset1.head(54), x='Continent', y='TotalTests', hover_data=['Country/Region'], color='TotalTests', size='TotalTests', size_max=80, log_y=True).show()
px.scatter(dataset1.head(30), x='Country/Region', y='TotalCases', hover_data=['Country/Region'], color='Country/Region', size='TotalCases', size_max=80, log_y=True).show()
px.scatter(dataset1.head(30), x='Country/Region', y='TotalDeaths', hover_data=['Country/Region'], color='Country/Region', size='TotalDeaths', size_max=80).show()
px.scatter(dataset1.head(30), x='Country/Region', y='Tests/1M pop', hover_data=['Country/Region'], color='Tests/1M pop', size='Tests/1M pop', size_max=80).show()
px.scatter(dataset1.head(30), x='TotalCases', y='TotalDeaths', hover_data=['Country/Region'], color='TotalDeaths', size='TotalDeaths', size_max=80).show()
px.scatter(dataset1.head(30), x='TotalCases', y='TotalDeaths', hover_data=['Country/Region'], color='TotalDeaths', size='TotalDeaths', size_max=80, log_x=True, log_y=True).show()
px.scatter(dataset1.head(30), x='TotalTests', y='TotalCases', hover_data=['Country/Region'], color='TotalTests', size='TotalTests', size_max=80, log_x=True, log_y=True).show()

# ✅ Time-series bar charts
px.bar(dataset2, x="Date", y="Confirmed", color="Confirmed", hover_data=["Confirmed", "Date", "Country/Region"], height=400).show()
px.bar(dataset2, x="Date", y="Confirmed", color="Confirmed", hover_data=["Confirmed", "Date", "Country/Region"], log_y=True, height=400).show()
px.bar(dataset2, x="Date", y="Deaths", color="Deaths", hover_data=["Confirmed", "Date", "Country/Region"], log_y=False, height=400).show()

# ✅ Country-specific time series: US
df_US = dataset2.loc[dataset2["Country/Region"] == "US"]
px.bar(df_US, x="Date", y="Confirmed", color="Confirmed", height=400).show()
px.bar(df_US, x="Date", y="Recovered", color="Recovered", height=400).show()
px.line(df_US, x="Date", y="Recovered", height=400).show()
px.line(df_US, x="Date", y="Deaths", height=400).show()
px.line(df_US, x="Date", y="Confirmed", height=400).show()
px.line(df_US, x="Date", y="New cases", height=400).show()
px.bar(df_US, x="Date", y="New cases", height=400).show()
px.scatter(df_US, x="Confirmed", y="Deaths", height=400).show()

# ✅ Choropleth maps
px.choropleth(dataset2, locations="iso_alpha", color="Confirmed", hover_name="Country/Region", color_continuous_scale="Blues", animation_frame="Date").show()
px.choropleth(dataset2, locations='iso_alpha', color="Deaths", hover_name="Country/Region", color_continuous_scale="Viridis", animation_frame="Date").show()
px.choropleth(dataset2, locations='iso_alpha', color="Recovered", hover_name="Country/Region", color_continuous_scale="RdYlGn", projection="natural earth", animation_frame="Date").show()

# ✅ Region bar animation
px.bar(dataset2, x="WHO Region", y="Confirmed", color="WHO Region", animation_frame="Date", hover_name="Country/Region").show()

# ✅ WordClouds from dataset3
# Drop NA
sentences = dataset3["Condition"].dropna().tolist()
word_text = ' '.join(sentences)
plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(word_text))
plt.axis("off")
plt.show()

# Second WordCloud
column2 = dataset3["Condition Group"].dropna().tolist()
column2_str = " ".join(column2)
plt.figure(figsize=(10,10))
plt.imshow(WordCloud().generate(column2_str))
plt.axis("off")
plt.show()
