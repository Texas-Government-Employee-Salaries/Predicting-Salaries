![Image of logo](https://github.com/Texas-Government-Employee-Salaries/Predicting-Salaries/blob/master/logo.png)
# Capstone Project: Houston, We Have a Pay Gap.
by Mariam Naqvi, Christopher Mayorga, Desiree McElroy and Forrest McCrosky

## [Predicting Texas Government Employee Salaries](https://public.tableau.com/app/profile/desiree.mcelroy/viz/HoustonWeHaveaPayGap/HoustonWeHaveaSalary_)

## Project Description

Despite being a touchy subject, salary figures are a critical data point for organizations who value diversity, equity, and inclusion. Using data acquired from The Texas Tribune, our goal is to create a regression model that predicts a government employee’s annual salary based on demographic information. In doing so we will provide a methodology for companies and organizations who seek to analyze their own salary data and attain pay equity.

## Project Goals

1. Create scripts to perform the following:
 - acquisition of data
 - preparation of data
 - exploration of data

2. Perform statistical analysis to test hypotheses

3. Build and evaluate Regression models to predict annual salary for Texas State employees.


## Business Goals

* Discover drivers of annual salary and the distribution of annual salary based on features. 
* Reveal any inequities among various races and genders across salaries, elected officials, and director positions.
* Perform modeling, analysis and testing to verify the performance of a prediction model using linear regression.

## Initial Hypotheses
*Hypotheses 1:* I rejected the null hypotheses; 
* Confidence level = 0.99
* Alpha = 1 - Confidence level = 0.01
* H<sub>0</sub>: There is no correlation between tenure years and annual salary.
* H<sub>1</sub>: There is a correlation between tenure years and annual salary.

*Hypotheses 2:* I rejected the null hypotheses; 
* Confidence level = 0.99
* Alpha = 1 - Confidence level = 0.01
* H<sub>0</sub>: Texas Government employees and whether or not they fall in the BIPOC category is idependent of their annual salary
* H<sub>1</sub>: Texas Government employees and whether or not they fall in the BIPOC have a significant difference in their annual salaries

## Project Findings
* Employees with director titles have a much higher salary than those without director titles.
* There is a major disparity between genders among elected officials and employees with director titles.
* The white race is over-respresented in elected officials and director titles as compared to ethnicities across all government positions.
* Salary distribution graphs also showed an equal distribution amongst the various races up to a certain amount. However as the salary rate trended higher, bipoc ethnicity totals as well as the count for women, lowered.

## Data Dictionary
Please use this data dictionary as a reference for the variables used within in the data set.

| Feature       | Data type     | Description     |
| :------------- | :----------: | -----------: |
|  agency_id | int64   | Unique agency identifier    |
|  agency  | object | Identifier for agency or department |
|  lastname	| object	| Employee's last name |
| firstname	| object	| Employee's first name | 
| title	| object	| Employee's job title |
| race	| object |	Employee's race. Options are White, Hispanic, Black, Asian,  Native American, and Other|
| sex	| object |	Employee's sex. Options are male or female. |
| emptype	| object |	Whether an employee is a) classified or unclassified, b) regular or temporary, c) part-time or full-time |
| hire_date	| datetime64 |	When the employee was hired |
| hours_worked	| float64	| How many hours per week an employee worked |
| monthly_salary	| float64	| An employee's monthly salary | 
| annual_salary	| float64 |	An employee's annual salary |
| is_female	|int64|	An employee's sex encoded. 0=Male, 1=Female |
| is_white |int64|	An employee's race encoded. 0=Not White, 1=White |
| is_hispanic|	int64	|An employee's race encoded. 0=Not Hispanic, 1=Hispanic |
| is_black	|int64	|An employee's race encoded. 0=Not Black, 1=Black |
| is_BIPOC	|int64|	An employee's race encoded. 0=White, 1=Hispanic, Black, American Indian, Asian, or Other |
| race_encoded	|int64	|An employee's race encoded. 0=Am Indian, 1=Asian, 2=Black, 3=Hispanic, 4=Other, 5=White |
| tenure_months	|int64	|An employee's tenure in months. Calculated by subtracting 7/1/2021 and the hire date. 7/1/2021 was the date this data was most recently updated. |
| tenure_years	|float64	|An employee's tenure in years |
| is_elected	|int64	|0=Employee was not elected. 1=Employee was elected |
| is_director	|int64	|0=Employee does not contain "director" in their title. 1=Employee contains "director" in their title. |
| is_unclassified	|int64	|0=Employee has access to classified information. 1=Employee does not have access to classified information. |
| is_parttime	|int64	|0=Employee is full-time. 1=Employee is part-time. |
| tenure_years_bins	|int64	|Assigns an employee to a "bin" based on their tenure. 1=0-5 years. 2=5-10 years. 3=10-20 years. 4=More than 20 years. |
| 0-5_years	|int64	|An employee's bin encoded. 0=Tenure does not fall within 0-5 years. 1=Tenure falls within 0-5 years. |
| 5-10_years	|int64	|An employee's bin ecnoded. 0=Tenure does not fall within 5-10 years. 1=Tenure falls within 5-10 years. |
| 10-20_years	|int64	|An employee's bin encoded. 0=Tenure does not fall within 10-20 years. 1=Tenure falls within 10-20 years.|
| >20_years	|int64	|An employee's bin encoded. 0=Tenure is less than 20 years. 1=Tenure is greater than 20 years. |
***

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
This is accomplished via the python script named `acquire.py`. 
* The function `get_texas_data` will use pandas to read a csv found on The Texas Tribune's website.
* The `get_data_summary` function will present a number of data-set metadata, including the following:
  * The number of rows/columns in the data set
  * The number of missing values
  * Basic information about the data
  * Summary stats for the data and value counts
  * Listings of each category and relative proportions

### 3. Prepare
This functionality is stored in the python script `prepare.py`. 
* `explore_univariate` will perform the following actions:
1. Examine individual distributions of data and identify outliers
2. perform univariate analysis, by generating bar plots for each categorical variable, as well as box plots and histograms for quantitative variables
* `prepare_tex` performs the following actions:
1. Lowercases the capital columns and renames all the abbreviated column to a more human readable format. 
2. The function also trims leading and trailing white space on all the string values for the object columns.
3. Check for duplicate rows in the data set. If duplicates are detected, they are removed and appropriate log messages are returned
4. Check for clerical errors in the data set - several such cases were identified and addressed as follows:
	* Three employees had a hire date listed as 2069. Since it was such a low number, we dropped these emplooyees 
5. Drop unnecessary columns such as `jobclass`, `mi`, `rate`, `statenum`, `duplicated`, `multiple_full_time_jobs`, `combined_multiple_jobs`, `summed_annual_salary`, `hide_from_search`. 
6. Renamed columns for ease of workflow
* `create_features` performs the following actions:
1. One hot encodes for sex: `is_female` column
2. One hot encodes for top three races: `is_white`, `is_black`, `is_hispanic` columns
3. One hot encodes for whether or not someone is a Black, Indigenous, Person of Color (BIPOC): `is_BIPOC` column
4. Creating `tenure_months` and `tenure_years` columns by taking 7/1/2021 - `hire_date`. The data was retrieved by the Texas Tribune on 7/1/2021.
5. Creating a `is_elected` categorical column using a conditional clause. Elected officials include: Governor, Lieutenant Governor, Attorney General, Justices, Comptroller of Public Accounts, Commisioners, and state legislators.
6. Creating a `is_director` categorical column of whether or not someone is a director of their department.
7. Creating a `is_unclassified` categorical column using a conditional clause. 
8. Creating a `is_parttime` categorical column using a conditional clause.
* Moreover, we opted not to remove outliers from the train split, as there would potentially be outliers in the validate/test sets. This could negatively impact the model's performance.
* `make_bins_and_feats` performs the following actions:
1. Uses `pd.cut` to create 4 bins out of the `tenure_years` column
2. Prints out the value counts of each bin
3. Uses one hot encoding to create categorical columns for each of the bins (`0-5_years`, `5-10_years`, `10-20_years`, `>20_years`)
* `split_data` performs the following actions:
1. Split the data into 3 datasets - train/test/validate - used in modeling
	* Train: 56% of the data
	* Validate: 24% of the data
	* Test: 20% of the data
* `min_max_scale` performs the following actions:
1. Creates a scaler object by fitting it onto the train dataset
2. Scales X_train, X_validate, and X_test as arrays
3. Converts arrays to dataframes

### 4. Explore
* This functionality resides in the `explore.py` file, which provides the following functionality:
  1. Perform bivariate analysis, by generating bar plots for categorical variables, as well as scatter plots for quantitative variables
  2. Perform multivariate analysis by generating scatter plots of each continuous variable against the target variable, by each categorical variable  
* Performed T-tests, correlation tests, and chi-squared tests to test our initial hypotheses

### 5. Model
* Feature Selection:
	* Used Correlation (of predictors with the target variable) and RFE to select the top 5 features to include in the model
	* The following were selected:
		* `is_female`, `is_hispanic`, `is_black`, `is_BIPOC`, `is_director`, `is_unclassified`, `is_parttime`, `0-5_years`, `5-10_years`, `>20_years`
* Generate a baseline, against which all models will be evaluated
	* The baseline was calculated to have an RMSE of $26,448.16; each of the models was evaluated against this baseline value
* Compare the models against the baseline and deduce which has the lowest RMSE and highest R-squared value
* Fit the best performing model on test data
* Create visualizations of the residuals and the actual vs predicted distributions

### 6. Deliver
* [Present findings via Tableau story](https://public.tableau.com/app/profile/desiree.mcelroy/viz/HoustonWeHaveaPayGap/HoustonWeHaveaSalary_)

## To recreate
Simply clone the project locally and follow steps outlined in this README.

Next, open the Jupyter notebook titled `final_notebook` and execute the code within. 

## Takeaways
During the analysis process, I made use of the following regression models:
1. OLS Regression
2. Lasso + Lars
3. Tweedie Regressor GLM
4. Polynomial Regression (2nd degree)

Our results indicated that the Polynomial Regression model provided the highest R-squared of 31% and the lowest RMSE of $21,301.19 USD. This beat the baseline RMSE of $62,087 and R-squared of 0. Our model outperforms the baseline by 19.5%. 

The tenure and race variables were found to be the best drivers of annual salary.

## Next Steps
If we had more time, we would:
* explore other scaling methods
* collect more data related to higher annual salaries
