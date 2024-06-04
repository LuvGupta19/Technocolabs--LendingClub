# Technocolabs--LendingClub

<h2>Introduction</h2>
In the last few years, applying for different types of loans through online peer-to-peer lending platforms such as the LendingClub is raising. "LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California."
On the LendingClub platform, people invest on other people loan through an online secured system. On these types of platforms, in the most cases, the main criteria of giving loans to costumers is solely based on their credit scores, so that a customer with lower credit score (more risky) get higher rate and customer with higher credit score (less risky) get lower interest rate for their loan. Obviously, from the investor point of view, the loans with higher interest rate are more attractive due to their higher return of investment. However, it also has high risk of being not returned at all.

So, investing on "Bad loans" or charged-off, which means you loose your asset, is more worse than loosing an opportunity to gain more profit. The machine learning/deep learning model that could predict which of the high interest loans are more likely to be returned, would bring added value by minimizing the associated risks. Also, using other factors along with credit score may help us to identify the high risky loans and minimize the investors loss of money more accurately.


<h2>Our Goal</h2>
In this capstone project, We will going to work on LendingClub Dataset obtained from Kaggle and the goal is to try find a better prediction model to prevent investing on '"bad loans". To do that, First, going to implement some data engineering and preprocessing on LendingClub dataset to prepare data for analysis and modeling. Second, we hvae to apply explanatory data analysis (EDA) to investigate the features. At the end, we use preprocessed data on LendingClub loans labeled on whether or not the borrower defaulted (charged-off) to develop a model and predict whether or not a borrower will pay back their loan. This way in the future when we get a new potential customer who assigned with higher interest loan, we can assess whether or not they are likely to pay back the loan.


<h2>Dataset</h2>
There are different datasets for LendingClub loans in Kaggle but some of them missed some features. Thus, I used LendigClub Dataset possessing almost all features including FICO scores. This dataset contains more than several millions data and because, here, I only use a normal laptop to analyze and model this dataset, thus, I only selected the loans issued in 2018 (almost 0.5 million data) to reduce the processing time. After the first step of preprocessing, we have 109 features and around 0.5 million of data. TRY IT!

Before starting explanatory data analysis, it is a good idea to do some data engineering and preprocessing to prepare data for analysis and modeling.

**DataSet Link -** https://www.kaggle.com/wordsforthewise/lending-club/download
