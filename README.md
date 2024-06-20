### OpenDoor Assignment 

#### Analysis in Notebook firstAnalysis.ipynb

So first, I started off by reading the data. My first impression was this is a good, clean dataset. 

My first step was to look at the missing values, see if there are NaN values and plot the distribution for the columns in the dim_homes data

One of the first things I observed about the target variable was the it is an imbalaced dataset with respect to it. About 84% of offers not accepted and 16% of offers accepted. One of the ideas I has was to use SMOTE and imbalanced-learn to mitigate this issue. 

I played around merging fct_offers and dim_homes tables to generate a larger dataset which was then use for modelling. Since this is a clear case of classification, I used a combination of Logistic Regression, Support Vector Machines, Random Forest Classification, and XGBoost to model the data. I observed precison and recall suffered badly and suspect this is because of the imbalance of the class and also probably because of it was a rough model and did not undergo and preprocessing. One thing to note is that the F1-Score is still high.  

My take away from this more that I could understand the different tables and columns and how I can use them for analaysis later.


#### Analysis in Notebook secondAnalysis.ipynb 

In addition to the analysis in the first notebook, I observed that the dataset has offers sent out only to a fraction of the homes that were sent mails to. It makes logical sense. Also, some homes were sent multiple offers (24 in number). 

Also, I tried to create better features, namely home_age, years_since_last_transaction, value_per_sqft from the existing features which is more logical from buying a home point of view. 

Since market and home_type are categorical in type , I felt using one-hot-encoding for these two features is a better option.

Here I tried an inner join of the dim_homes with the fct_offers to creare a new dataset and then modelled it. 

However, I ran into the error as I did not know how I could predict the top 10000 most rewarding homes. Also, since my data is only about 7k rows it showed my approach was flawed. Also, I have not yet used the fct_mailers table yet !! 


#### Analysis in Notebook weightedAnalysis.ipynb

My first step was to analyse how different table obtained from joining the fct_mailers and fct_offers tables on the home_id. It clearly showed that they cover all the home_ids. 

I then also create a new feature (I did not use this here but would idealy want to use it), time_diff to get the difference between the offer_date and mail_date. 

So, my idea of getting the top 10000 home_id to send fliers to is as follows: 
    - First left join the fct_mailers, fct_offers on home_id and then left join the merged dataframe with the dim_homes tables on home_id
    - I then created better features, namely home_age, years_since_last_transaction, value_per_sqft from the existing features which is more logical from buying a home point of view ( simiar to approach in secondAnalysis.ipynb notebook)
    - To create the train and predict (test) dataset I did the following.
    1) The data with offer_accepted ( True and False ) became the train dataset and has 7482 rows. 
    2) The data with no data in the offer_accepted column is where I had to predict the top 10000 home_ids. 

I contemplated on a few algorithms namely -  Stratified Sampling ensures that the training data is representative of the overall population, Active Learning iteratively improves the model by focusing on uncertain predictions, Constraint-Based Optimization applies specific constraints to ensure a diverse and high-quality selection of homes.

I first tried to overcomplicate my solution by doing a weighted average of predictions from all the three methods. 

Given this is the first time I am using these algorthims and the computational resouces I posses, I took a step back and ended up choosing Constraint-Based Optimization which I believe is a good way of dealing with the problem statement. 

My algorithm ran for close to 10 hours to give an output !! I evaluted this by dropping the rows that were given offers in the dim_homes table and then sorting the dim_homes by est_home_value and comparing the two lists with the mean, median, and standard deviation.

Things that could have been done: 

1) Model Optimization - using a combination of Stratified Sampling, Active Learning, and Constraint-Based Optimization. 
2) A weighted optmization of the combination of above algorithms
3) Using KNN to interpolate the Zero values for offer_amount in the test data
4) Additional ways of evaluating the process like using different metrics, Pearson's Correlation Coefficient, Statistical Tests. 

I am sure there are more that are possible but this is my analysis. 