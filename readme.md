# Capstone Project: Houston, we have a salary.
by Mariam Naqvi, Christopher Mayorga, Desiree McElroy and Forrest McCrosky

## Predicting Texas Government Employee Salaries

## Project Description

## Project Goals

1. Create scripts to perform the following:
 - acquisition of data
 - preparation of data
 - exploration of data

2. Perform statistical analysis to test hypotheses

3. Build and evaluate Regression models to predict annual salary for Texas State employees.

4.
## Business Goals

* Discover drivers of annual salary and the distribution of annual salary based on features. 
* Perform modeling, analysis and testing to verify the performance of a prediction model using linear regression.

## Initial Hypotheses
*Hypotheses 1:* I rejected the null hypotheses; 
* Confidence level = 0.99
* Alpha = 1 - Confidence level = 0.01
* H<sub>0</sub>: 
* H<sub>1</sub>: 

*Hypotheses 2:* I rejected the null hypotheses; 
* Confidence level = 0.99
* Alpha = 1 - Confidence level = 0.01
* H<sub>0</sub>: 
* H<sub>1</sub>: 

## Project Planning

The overall process followed in this project, is as follows:

1. Plan
2. Acquire
3. Prepare
4. Explore
5. Model
6. Deliver

### 1. Plan
* Create a list of tasks to complete in the <a href="https://trello.com/b/izaNBd0G/capstone-project">Trello Board</a>
* Perform preliminary examination of the dataset
* Collect database details (connection information and credentials)

### 2. Acquire
* This is accomplished via the python script named “acquire.py”. The script will use credentials (stored in env.py) to collects data using a SQL query from the following tables:
	1. 
	2.  
	3. 
	* 
		*  

### Columns Selected from the Original Data 
- 
 - 
- 
- 

*
* Finally, the get_data_summary() function will present a number of data-set metadata, including the following:
  * The number of rows/columns in the data set
  * The number of missing values
  * Basic information about the data
  * Summary stats for the data and value counts
  * Listings of each category and relative proportions

### 3. Prepare
* This functionality is stored in the python script "prepare.py". It will perform the following actions:
1. Examine individual distributions of data and identify outliers
* perform univariate analysis, by generating bar plots for each categorical variable, as well as box plots and histograms for quantitative variables
3. Check for duplicate rows in the data set. If duplicates are detected, they are removed and appropriate log messages are returned
4. Check for nulls in the data set - several such cases were identified and addressed as follows:
	* 
	* 
	    * 
	    * 
	* 
		* 
5. 
6. 
7. 
8. Attempt to remove the outliers using an IQR of 1.5 - although this did bring some distributions closer to normal, this reduced the correlation of predictors with the target variable. As a result, we proceeded without this process and instead made use of a Robust Scaler to reduce the effect of outliers.
	* Moreover, we opted not to remove outliers from the train split, as there would potentially be outliers in the validate/test sets. This could negatively impact the model's performance.
9. Split the data into 3 datasets - train/test/validate - used in modeling
	* Train: 56% of the data
	* Validate: 24% of the data
	* Test: 20% of the data

### 4. Explore
* This functionality resides in the "explore.py" file, which provides the following functionality:
  1. Perform bivariate analysis, by generating bar plots for categorical variables, as well as scatter plots for quantitative variables
  2. Perform multivariate analysis by generating scatter plots of each continuous variable against the target variable, by each categorical variable  
* Performed T-tests and correlation tests to test my initial hypotheses

### 5. Model
* Feature Selection:
	* Used Correlation (of predictors with the target variable) and RFE to select the top 5 features to include in the model
	* The following were selected:
		* '', '', '', '', ''
* Generate a baseline, against which all models will be evaluated
	* The baseline was calculated to have an RMSE of ; each of the models was evaluated against this baseline value
* Compare the models against the baseline and deduce which has the lowest RMSE and highest R-squared value
* Fit the best performing model on test data
* Create visualizations of the residuals and the actual vs predicted distributions

### 6. Deliver
* Present findings via PowerPoint slides

## To recreate
Simply clone the project locally and create an env.py file in the same folder as the cloned code. The format should be as follows:

```
host = ‘DB_HOST_IP’
user =  ‘USERNAME’
password = ‘PASSWORD’

def get_db_url(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
```
    
In the above code, replace the `host`, `user` and `password` values with the correct Database Host IP address, Username and Password.

Next, open the Jupyter notebook titled “final_report_zillow” and execute the code within. 

## Takeaways
During the analysis process, I made use of the following regression models:
1. OLS Regression
2. Lasso + Lars
3. Tweedie Regressor GLM
4. Polynomial Regression

My results indicated that the Polynomial Regression model provided the highest R-squared of 44% and the lowest RMSE of 463791. This beat the baseline RMSE of 620877 and R-squared of -22%.

The square footage, location (latitude and longitude) and number of bedrooms were found to be the best drivers of tax value.

## Next Steps
If I had more time, I would:
* add more features to the models - garage, basement, pool
* explore other scaling methods
* collect more data related to higher-property values
