import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV

# Attempt to import libraries and handle any import errors
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as e:
    st.error(f"An error occurred while importing libraries: {e}")
    st.stop()

# Set up the main title and sidebar
st.sidebar.markdown("<h1 style='color:Orange;'>Table of Contents</h1>", unsafe_allow_html=True)

pages = ["Overview", "Data exploration", "Data Visualization","Data Pre-processing" , "Machine Learning", "Predictions countries", "Summary"," References" ]
page = st.sidebar.radio("Go to", pages)


st.sidebar.markdown("""
    <h2 style='font-family:Calibri (Body); color:Orange;'>Project Team Members</h2>
    <p style='font-family:Calibri (Body); color:blue;'><strong>1. Leonie E. Fickinger</strong> 
    <a href='https://www.linkedin.com/in/leonie-ederli-fickinger-424a24117/' style='color:blue;'>
    <img src='LI-In-Bug.png'>
    </a></p>
    <p style='font-family:Calibri (Body); color:blue;'><strong>2. Marta Blazsik</strong> 
    <a href='https://www.linkedin.com/in/martablazsik' style='color:blue;'>
    <img src='LI-In-Bug.png'>
    </a></p>
    <p style='font-family:Calibri (Body); color:blue;'><strong>3. Nitin Zimur</strong> 
    <a href='https://www.linkedin.com/in/nitin-z-9a662019/' style='color:blue;'>
    <img src='LI-In-Bug.png'>
    </a></p>
    """, unsafe_allow_html=True)

# Load the datasets
@st.cache_data
def load_data():
    df1 = pd.read_csv('world-happiness-report-2021.csv')
    df2 = pd.read_csv('world-happiness-report.csv')
    return df1, df2

@st.cache_data
def merge_data(df1, df2):
    # Renaming columns in df2 to match df1
    new_column_names_df2 = {
        'Life Ladder': 'Ladder score',
        'Log GDP per capita': 'Logged GDP per capita',
        'Healthy life expectancy at birth': 'Healthy life expectancy'
    }
    df2 = df2.rename(new_column_names_df2, axis=1)
    
    # Adding year to df1
    df1["year"] = 2021

    # Merging the two data frames
    merged_df = df1.merge(right=df2, on=['Country name', 'year', 'Ladder score', 'Logged GDP per capita',
                                         'Social support', 'Healthy life expectancy',
                                         'Freedom to make life choices', 'Generosity',
                                         'Perceptions of corruption'], how='outer')
    
    merged_df = merged_df.reset_index(drop=True)
    return merged_df

df1, df2 = load_data()
merged_df = merge_data(df1, df2)

# Data preparation
X = df1[['Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
y = df1['Ladder score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Display an image
if page == pages[0]: 
    st.markdown("<h1 style='text-align: left;'>World Happiness Project</h1>", unsafe_allow_html=True)
    st.write("")
    st.image("WHR1.jpg", caption='', use_column_width=True)
    st.write("""This project analyses data informing the World Happiness Reports with the goal of predicting the five happiest and the five least happy countries for the year 2022. To this end a machine learning model was trained on socioeconomic data and perceptions collected during a time series 2005 to 2021 across countries.""") 
    st.write("""The World Happiness Report is a partnership of Gallup, the UN Sustainable Development Solutions Network, the Oxford Wellbeing Research Centre, and the WHR’s Editorial Board. Its aim is to inform government action and policies as well as raising awareness on happiness and well-being in general. The yearly released Reports present and analyse world happiness scores across countries in a ranking format. The Gallup World Poll - a worldwide yearly survey on life evaluation questions – is one of the main underlying data sets supporting the rankings. Besides life expectancy data from WHO and GDP data, the Gallup World Poll serves as the underlying data source of this project.""") 

# Data exploration Page

if page == pages[1]:


    st.write("### Relevance: Target and Feature Variables")
    
    st.write("""
    To predict the happiness scores of countries, six feature variables were selected.
    
    **Target Variable:**
    
    **Ladder score:** Happiness score representing subjective well-being. It is the national average response by individuals in a country to the question of own quality of life which they were able to rate along a ladder ranging from 0 to 10.
    """)
    
    st.write("""
    ### Feature Variables:
    
    **1. Logged GDP per capita:** GDP per capita is a measure of the economic output per person in a specific geographical area, usually a country.
    
    **2. Social support:** This variable represents an opinion of individuals about their own situation regarding how much they can rely onto social support (e.g., family, friends) in challenging situations. The Social Support value can vary between 0 and 1 and it's a national average of all individuals based on binary responses (0 or 1).
    
    **3. Healthy life expectancy:** This variable represents healthy life expectancies at birth, based on data from the WHO and original data.
    
    **4. Freedom to make life choices:** This variable represents the national average of responses to the question “Are you satisfied or dissatisfied with your freedom to choose what you do with your life?”. Values can vary between 0 and 1.
    
    **5. Generosity:** This variable represents a quantification of how much more or less generous a country’s population is compared to what is expected when considering the country's GDP per capita. A negative value indicates that the country's population seems to have a less generous donation behavior than what the GDP per capita variable would predict.
    
    **6. Perceptions of corruption:** This variable indicates the national average perception of whether corruption is widespread throughout the government and within businesses. As the responses for the two questions used for this indicator are binary, the values of this variable can only range between 0 and 1.
    """)
    
   

    st.write("""
### Indicative Variables:

**1. Country name:** This variable indicates the country to which the values of a row are associated with.

**2. Year:** This variable indicates the year to which the values of a row are assigned to.
""")

    st.write("""
### Data Sources:

The data input for this project consists of three main data sources which were filtered according to the project’s needs. This means, only the columns containing the six selected feature variables, the target variable and the columns indicating the year and the country of these data sets were of interest for the project. The three data sources are:

**1. world-happiness-report-2021.csv:** This file contains world happiness scores for the year 2021 for 149 countries, including the corresponding feature variables taken from the Gallup World Poll. This file also contains data presenting further analysis on the ladder score (e.g., contribution effects of explanatory variables on the happiness score). This data set contains no missing values.

**2. world-happiness-report.csv:** This file contains data from the Gallup World Poll for 166 countries over the timespan 2005 to 2020. It also includes data on positive and negative emotions collected by the Gallup World Poll. This data set contains missing values in the columns representing the feature variables.

**3. DataForTable2.1WHR2023.xls:** This file contains data from the Gallup World Poll for 166 countries over the timespan 2005 to 2022. It also includes data on positive and negative emotions collected by the Gallup World Poll. This data set contains no missing values for the ladder score.
""")

    

    st.write("""In the following steps, the first two datasets were merged resulting in a dataset containing data over the timespan 2005 to 2021 and preprocessed according to visualization and modelling needs.The last data set was only used for retrieving 2022 data and applying the selected machine learning model on it""")

    st.write("The following data set represents the data set resulting from merging the first two data sources on the common columns only.")

    st.image("Merged_Dataset.jpg", use_column_width=True)

# Data Visualization Page

if page == pages[2]: 

    st.write("### Visualization")

    st.write("""
After studying all dimensions and measurements thoroughly included in the data frames and conducting basic pre-processing steps, we kept on familiarising ourselves with the data through visualisation graphs.

First, the team focused on the year 2021 being the most recent year in the given date range, but also because it offered some additional information, for example the Regional Indicator.

To have a first impression of how the ladder score value is distributed across countries in 2021, the project team computed a map showing the geographical distribution of the happiness scores using the Plotly express library (below). This library allows plotting a world map using the px.choropleth in which the countries covered by the dataset (149 countries) are coloured based on the ladder score value. Benefitting from Plotly’s interactive feature, by hovering your mouse over the countries, you can see their name and ladder score value. 

             
Countries not covered by the world_happiness-2021 dataset are shown without colour (i.e. white).  
""")          
# Create the Plotly choropleth map
    fig = px.choropleth(
    df1,
    locations='Country name',
    locationmode='country names',
    color='Ladder score',
    hover_name='Country name',
    color_continuous_scale='RdBu',
    title='Figure 1: Ladder Score Distribution Across Countries in 2021'
    )

    fig.update_geos(showframe=False)  # Hide the frame around the map
    fig.update_layout(coloraxis_colorbar=dict(title='Ladder Score'))
    fig.update_traces(zmin=0, zmax=8)
    fig.update_layout(legend_title_text='Happiness Score')

# Display the Plotly figure in the Streamlit app
    
    st.plotly_chart(fig)             



    st.write("""
The map shows some clear distribution patterns. OECD countries seem to occupy the top range of the ladder scores, meanwhile Africa and the Middle East tend to be rather at the lowest end. The very dark blue and dark red colours - representing the maximum and the minimum ladder score values - seem to be associated with only a few countries.

Additionally, we can observe the grouping of countries by region, certain neighbouring country groups being in the same range of the Ladder score. 

The column Regional Indicator was only present in the 2021 data frame, and to answer our assumption, if the Region could be an indicator of the range in the Ladder score, we created a graph displaying its distribution by Region, with boxplots: 

    """)         

    fig_boxplot, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=df1, y='Regional indicator', x='Ladder score', hue='Regional indicator', ax=ax, dodge=False, order=df1['Regional indicator'].sort_values().unique())
    ax.set_title("Happiness Score by Region", size=20)
    # Remove legend if it exists
    if ax.legend_:
        ax.legend_.remove()

# Display the box plot in the Streamlit app
    st.pyplot(fig_boxplot)


    st.write("""
Even though the length of the distribution range differs a lot depending on the Region, based on this graph, our assumption seems correct (however, we did not statistically analyse it further and relied only on the visual graph). 

Now, moving on from the 2021 data set to the entire data set, let’s take a look at the distribution and evolution of variables through the years depending on your choice, focusing on the main variables, namely: Ladder score, Social support, Logged GDP per capita, Healthy life expectancy, Freedom to make life choices, Generosity and Perceptions of corruption:
    """)

    fig_violinplot, ax = plt.subplots(figsize=(10, 7))
    
    selected_dimension = st.selectbox("Select a dimension:", ["Ladder score", "Logged GDP per capita",
                                         "Social support", "Healthy life expectancy",
                                         "Freedom to make life choices", "Generosity",
                                         "Perceptions of corruption"])

    sns.violinplot(x="year", y=selected_dimension, data=merged_df)
    plt.xticks(rotation=90)

    st.pyplot(fig_violinplot)

    st.write("""
Meanwhile the temporal factor was not our main focus in the project, it’s interesting to see how these values change over time in the world, but to understand the underlying issues resulting in this evolution requires a deeper analysis. 

Next, our team studied correlations between the main variables for the whole time frame 2005-2021. 

Scatterplots offer a great visual representation of correlations among variables: 

    """)
         
    st.image("Scatterplot2.png", caption='Merged data set 2005-2021 with common columns', use_column_width=True)

    st.write("""
    To offer a better overview with exact numbers, we also created a Heatmap:   
    
""")
                
# Heatmap 2005-2021

    heatmap_table = merged_df[["Ladder score", "Logged GDP per capita", "Social support",
                         "Healthy life expectancy", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]]

    cor2 = heatmap_table.corr()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cor2, annot=True, ax=ax, cmap="coolwarm")
    plt.title("Heatmap 2005-2021")

# Display the heatmap in the Streamlit app
    st.pyplot(fig)


    st.write("""
There are strong positive correlations between variables such as GDP per capita, Social support, Life expectancy and the Ladder score. This suggests that countries with higher GDP per capita, stronger social support systems, and longer life expectancies tend to have higher happiness scores. A Pearson test also statistically confirmed the positive correlations of the Ladder score with GDP per capita and Healthy life expectancy.

On the other hand, the Perception of corruption negatively correlates with the Ladder score. 

Finally, we identified the five happiest and least happy countries over the entire time frame 2005-2021 by computing the average ladder score per country:
      
""")
    st.image("The Happiest and least Happy countries.jpg", use_column_width=True)

# Data pre-processing Page
if page == pages[3]:
    st.write("### Data Preprocessing and Merging Data Sets")
    st.write("""
    The data sets **world-happiness-report-2021.csv** and **world-happiness-report.csv** were merged with the aim to obtain a data set containing data for a continuous time series ranging from 2005 to 2021, focusing only on the columns of interest.
    Furthermore, the dataset was reduced to the minimum – meaning, only the feature and the target variables were kept and rows where more than two missing values of the feature variables are missing (i.e., at least 33% of the explanatory data missing) were deleted. As these columns all contain numerical values, no encoding was necessary.
    """)

    st.write("### Data Cleaning")
    st.write("""
    No duplicates or outliers were present in the merged data set. Yet, missing values were found and had to be imputed. To this end, the project team produced two different merged data set versions – one for visualization purposes, the other for machine modelling usage. For the latter, best practice was followed by imputing missing data only after splitting the data set into a train and a test set.
    
    **- Data imputation for visualization:** Imputation was carried out by substituting the missing data with the median of each respective column.
    
    **- Data imputation for machine modelling:** Imputation was carried out by substituting the missing numerical variable values with the median value of each country. In the case where no country values existed for the variable of interest, the median of the column (vs. the median of the country) was taken. For machine modelling the project team wanted to replace data in a way representative as much as possible for a country. Therefore, this approach was chosen over taking the median of each column.
    """)

    st.write("### Normalization")
    st.write("""
    Given our feature variables have different ranges, it is advisable to carry out normalization or standardization on those variables. As most values in the columns do not follow a normal distribution, the project team decided against standardization, which is suggested in case of normal distribution. Instead, normalization was carried out on the six feature variables using the Min-Max-Scaler. This choice was made despite the presence of values on the upper and lower extreme. Min-Max-Scaler, as opposed to e.g., robust scaler, is sensitive to outliers but these values represent extreme values (vs. outliers) and therefore should be considered as part of the representative range. The normalization step was carried out once the data set was split.
    """)




# Machine Learning Page
if page == pages[4]: 

    st.write("### Machine learning ")
    
    st.write("""
    ### Classification of the problem

    The Happiness Report project is handling a regression problem. This is because we are trying to predict a continuous numerical variable – the ladder score variable – based on socioeconomic data and perceptions (measured in numeric terms).

    ### Model and performance testing

    The team tried different machine learning models adequate for regression problems – namely, the Linear Regression Model, the Decision Tree Regressor Model and the Random Forest Model.

    Different performance metrics suitable for regression type problems were calculated to inform the selection of the most appropriate model: the scores (R2), the Mean Absolute Error (MAE), the Mean Squared Error (MSE) and the Root Mean Squared Error (RMSE) and the Cross Validation Score.
    """)
   

    st.image("Table Performance scores_white background.jpg", caption='Comparison of performace scores for three models', use_column_width=True)

    st.write("""Model selection""")
    st.write("""The Random Forest Model is chosen as the best model as it shows the best scores. Further optimization of the chosen Model was assessed, specifically feature selection and dimensionality reduction. However, both techniques did not contribute to a significant improvement of the model and was, hence, not implemented.
     """)
    
    ####Model selection[tbc]



# Predictions for Countries Page

if page == pages[5]: 
    st.write("### Predictions for Countries")

    st.write("""
             
At this point, since the random forest model got evaluated, it’s time to return to our initial goal to predict the five happiest and least happy countries for the year following our data, namely 2022, and see how well the model performs in practice.

We tested the random forest model on the 2022 data and generated predictions for the Ladder score: 
 """)

    st.image("Predictions1.jpg", caption='', use_column_width=True)

    st.write("""
Since the 2022 data got published containing the Ladder score, it gave us the opportunity to compare our results with the actual ladder scores and the lists of the 5 happiest and least happy countries: 
 """)

    st.image("Predictions2.jpg", caption='', use_column_width=True)

   
    st.write("""   
     **Conclusion**

Taking a closer look at the five happiest countries, we can see that four out of five countries were predicted correctly to be in the top five. 
Yet, only two out of four were predicted correctly in terms of the ranking position (i.e. index) - which are Finland as happiest country and Denmark as third happiest country.

Regarding the five least happy countries, our predictions turn out to be less reliable, considering that only two countries, namely Afghanistan and Sierra Leone were predicted correctly to be in the five least happy countries. 

Note that the model was not intended to predict the right ranking but to calculate the expected happiness scores, however, the prediction of the five happiest and least happy countries, - which was our main project goal, - derives from the approach trying to create a more practical interpretation of the data.
""")
                
   


# Summary Page
if page == pages[6]: 
    st.write("### Summary")
    st.write(""" 

The World Happiness Report is one of, if not the most well known happiness research, and our team could learn a lot about how happiness is measured, which factors are taken into consideration as influential factors and from a statistical point of view, how each of these factors contribute to the overall results. 

We gained valuable insights along the way during data exploration, visualisation and statistical methods.

The analysis we conducted and the machine learning project provided us with the results we hoped for, getting predictions for the happiness score for 2022 and 5-5 country names representing the happiest and least happy countries. 

Overall, we can conclude that the predictions on the top five happiest countries were better than for the five least happy countries. Yet, with only six correctly predicted countries out of 10 (i.e. 60%), and with only two countries predicted correctly in terms of rating positions out of ten (i.e. 20%), the results are not as satisfying as expected. 

The project team would advise to investigate whether optimising other models, such as Linear Model or Decision Tree Regressor, would lead to better results. 
""")
    

# Summary Page
if page == pages[7]: 
    st.write("### Resources")

    st.write("""

Brownlee, J. (2020, September 19). Hyperparameter Optimization With Random Search and Grid Search. Machine Learning Mastery. https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/.

DataScientest. (2024). All sprints. https://learn.datascientest.com.

Helliwell, John F., Huang, H., Wang, S., & Norton, M. (2021). Statistical Appendix 1 for Chapter 2 of World Happiness Report 2021. https://happiness-report.s3.amazonaws.com/2021/Appendix1WHR2021C2.pdf.

Plotly. (2024). Mapbox Choropleth Maps in Python. https://plotly.com/python/mapbox-county-choropleth/.

SA. (2024). Optimizing Models: Cross-Validation and Hyperparameter Tuning Guide. StackAbuse. https://stackabuse.com/optimizing-models-cross-validation-and-hyperparameter-tuning-guide/.

World Happiness Report. (2024a). World Happiness Report 2021. https://worldhappiness.report/ed/2021/#appendices-and-data.

World Happiness Report. (2024b). Data for Table 2.1. https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fhappiness-report.s3.amazonaws.com%2F2023%2FDataForTable2.1WHR2023.xls&wdOrigin=BROWSELINK
""")
    
    
