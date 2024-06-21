### OpenDoor Assignment 

#### Analysis in Notebook firstAnalysis.ipynb

First, I began by reading the data. My initial impression was that this is a good, clean dataset.

The first step was to examine the missing values, check for NaN values, and plot the distribution for the columns in the dim_homes data.

One of the first observations about the target variable was that it represents an imbalanced dataset. Approximately 84% of offers were not accepted, while 16% were accepted. One idea I had was to use SMOTE and imbalanced-learn libraries to mitigate this issue.

I experimented with merging the fct_offers and dim_homes tables to generate a larger dataset, which was then used for modeling. Since this is clearly a classification problem, I employed Logistic Regression, Support Vector Machines, Random Forest Classification, and XGBoost to model the data. I noticed that precision and recall suffered significantly, likely due to the class imbalance and the use of a rudimentary model. Additionally, the data was not preprocessed. It's worth noting that the F1-Score remained high despite these issues.

My main takeaway from this analysis was a better understanding of the different tables and columns, and how I could utilize them for future analysis.

#### Analysis in Notebook secondAnalysis.ipynb 

In addition to the analysis in the first notebook, I observed that offers were sent out to only a fraction of the homes that received mailings, which is logical. Moreover, some homes received multiple offers (24 in total).

I also attempted to create more meaningful features, such as home_age, years_since_last_transaction, and value_per_sqft from the existing features, which are more relevant from a home-buying perspective.

Given that market and home_type are categorical variables, I felt that using one-hot encoding for these two features would be a better approach.

Here, I performed an inner join of the dim_homes with the fct_offers to create a new dataset, which I then modeled.

However, I encountered an error as I was unsure how to predict the top 10,000 most rewarding homes. Additionally, since my dataset contained only about 7,000 rows, it revealed that my approach was flawed. Furthermore, I had not yet utilized the fct_mailers table!

#### Analysis in Notebook weightedAnalysis.ipynb

My first step was to analyze the different tables obtained by joining the fct_mailers and fct_offers tables on the home_id. This clearly showed that they cover all the home_ids.

I also created a new feature, time_diff (which I didn't use here but would ideally want to incorporate), to calculate the difference between the offer_date and mail_date.

My approach for identifying the top 10,000 home_ids to send fliers to is as follows:
    - First, left join the fct_mailers and fct_offers on home_id, then left join the merged dataframe with the dim_homes table on home_id.
    - I then created more meaningful features, namely home_age, years_since_last_transaction, and value_per_sqft from the existing features, which are more relevant from a home-buying perspective (similar to the approach in secondAnalysis.ipynb notebook).
    - To create the train and predict (test) datasets, I did the following:
    1) The data with offer_accepted (True and False) became the training dataset, containing 7,482 rows.
    2) The data with no values in the offer_accepted column is where I had to predict the top 10,000 home_ids.

I considered several algorithms, namely: Stratified Sampling (which ensures that the training data is representative of the overall population), Active Learning (which iteratively improves the model by focusing on uncertain predictions), and Constraint-Based Optimization (which applies specific constraints to ensure a diverse and high-quality selection of homes).

Initially, I attempted to overcomplicate my solution by using a weighted average of predictions from all three methods.

Given that this was my first time using these algorithms and considering the computational resources at my disposal, I decided to take a step back. I ultimately chose Constraint-Based Optimization, which I believe is an effective way of addressing the problem statement.

My algorithm ran for nearly 10 hours to produce an output! I evaluated this by dropping the rows that were given offers in the dim_homes table, then sorting the dim_homes by est_home_value and comparing the two lists using mean, median, and standard deviation.

Areas for improvement:

1) Model Optimization - using a combination of Stratified Sampling, Active Learning, and Constraint-Based Optimization.
2) A weighted optimization of the combination of the above algorithms.
3) Using KNN to interpolate the zero values for offer_amount in the test data.
4) Additional methods of evaluating the process, such as using different metrics, Pearson's Correlation Coefficient, and Statistical Tests.

I'm certain there are more possibilities, but this concludes my analysis.
