# Home Credit Default Risk : 16th ranking solution

This github project contains the solution for Home Credit Default Risk competition. It consists on predicting whether a client will pay his debt in time or will default.

For this, Home Credit provided us a very rich dataset, that contains the following tables :
- application_{train,test}.csv : These tables contains information about applications to get a credit. Example : income, credit amount, birthday, etc...
- bureau.csv : The information contained in this table are provided by Credit Bureau. Credit Bureau is an organism that gather historical informations about past credits and payments of the borrowers. This table contains informations about past credits that were recorded by Credit Bureau. Each credit is related to an application in application_{train,test} tables.
- bureau_balance.csv : Each row in this table is related to bureau table. It gives monthly details of payments for each credit that were reported to Credit Bureau.
- previous_application.csv : Each application in application_{train,test} is related to a set of previous applications that were provided by Home Credit to the same applicant. This dataset gives information on those past credits.
- POS_CASH_balance.csv : Monthly balance snapshots of previous POS (point of sales) and cash loans. This table is related to previous_application table.
- credit_card_balance.csv : Monthly details about credit cards operations that clients had with Home Credit. Each row in this table is related to a credit in previous_application table.
- installments_payments.csv : Details about installments that clients did to pay loans reported in previous_application table.

## The project pipeline

The project followed a process that is described in the following steps :

### Increasing model accuracy with feature engineering 
This project was started with my teammate [Toaru](https://www.kaggle.com/marrvolo) and we were focused mainly on increasing the model accuracy (that was calculated using Cross Validation Roc Auc Score) by finding relevent features. We tried all sort of aggregations on data and some divisions that could make the model more accurate. Then a list of features was created and we proceeded to feature selection process that helped us get more accuracy in the created model. 

### Using public solutions to be more competitive with more crowded teams
Then, as my teammate gave up on the competition after one month of working, I had to find more features that could allow me to be more competitive with crowded teams and reach a high final ranking. To do so, I used the script posted in this public [discussion](https://www.kaggle.com/c/home-credit-default-risk/discussion/62983) to give more diversity to my final models. I also added features from this [kernel](https://www.kaggle.com/scirpus/hybrid-jeepy-and-lgb-ii).

### Making training datasets from all features created and collected wisely
The features collected from public discussions and kernels were generated using genetic programming, so they were overfitting the training set. To limit that overfitting, I took all the features collected and splitted them into 4 different dataframes. Then I added to each dataframe the features that I created by hand. By doing so, I could have 4 different training sets. The reason behind doing that operation was intuitive for me : the split that I did allowed me to use the overfitting features only once or twice, this could limit the overfitting in my final stacking model and gave more weight to the models that overfitted less. Moreover, the features I created by hand were the ones I trusted most and using them in each model would give them a significant weight in my final stacking model.

### Stacking all the models 
To maximise the Cross Validation score, I trained each training dataframe with xgboost and lightgbm. Then, I stacked all of them using the Logistic Regression like in this [kernel](https://www.kaggle.com/eliotbarr/stacking-test-sklearn-xgboost-catboost-lightgbm).
After the stacking, I removed the models that made my CV score drop down. Then I got my final stacking model that I used to make the final submission.

## Getting Started

This code uses a stacking of 11 final models, so it took me several hours to run it completely. But if you still want to run it, here are the instructions :

### Prerequisites

Install the following dependencies :
- pandas
- numpy
- xgboost
- lightgbm
- scikit-learn

### Prepare input files

Download the datasets after accepting the rules from Kaggle website and put them in the input directory.

### Run the code

navigate to the project from the console, then run the command : python app.py

When the code will finish running, the final_submission.csv that contains the final submission will be present in the processed/ directory.

In the app.py script, each method in the main function generates results that are saved in the processed/ directory. So, if for some reason you had to halt the script from working, just comment the lines that launches the scripts whose results were previously saved. That could save some time.

## Authors

* [Toaru](https://www.kaggle.com/marrvolo)

* **Ramzi Bouyekhf** - [propower](https://www.kaggle.com/propower)

## Acknowledgments

* [scirpus](https://www.kaggle.com/scirpus)