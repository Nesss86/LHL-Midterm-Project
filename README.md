# LHL-Midterm-Project


## Project/Goals
The goal of this project was to leverage historical US flight data to determine whether a model could be built to accurately forecast flight delays. The dataset contained a number of quantitative and categorical features such as departure time, arrival time, arrival delay, distance, airline, origin and destination airport, among others. The goal was to distill the model to only relevant features and use these features to build a classification model. 

## Process

#### 1.Identified Target Dataset
The comprehensive data set was acquired from https://www.kaggle.com/datasets/usdot/flight-delays/data for US flights in 2015 and provided over five million rows of total data to work with. The breadth and depth of the dataset provided confidence that statistically significant results could be derived from the overall model. 
#### 2.Conduct EDA
##### Cleaned data and One-hot encoded target variables
``` python
#Check which columns have too many rows of empty data

flights.isna().sum()
```
<img src="images/EDA.png" alt="Tableau Desktop">

``` python
#Based on analysis above, we can remove the columns below as they are mostly blank and will delete the entire dataset of we clean all columns without data. Also removing columns such as "Airline", "Origin Airport", "Destination Airport", and "Tail Number" such those aren't numerical and cannot be part of EDA

columns_to_drop = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TAIL_NUMBER', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DEPARTURE_TIME']
flights.drop(columns = columns_to_drop, inplace = True)



#One-hot encode categorical variables airlines, origin airport, and destination airport

flights2 = pd.get_dummies(flights, columns=['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'])
#Check shape of new DataFrame to confirm one-hot encoding was successfully executed

flights2.shape
```


#### 3.Built Basic & Advanced Classification Models
```python
#Import 2015 flight data from CSV file

dtype_options = {'AIRLINE': 'object', 'TAIL_NUMBER': 'object', 'ORIGIN_AIRPORT': 'object', 'DESTINATION_AIRPORT': 'object'}
flights = pd.read_csv('Data/flights.csv', dtype = dtype_options)

#Create final feature set based on testing parameters for full model evaluation
flights_final = flights[['ARRIVAL_DELAY', 'SCHEDULED_DEPARTURE', 'DISTANCE', 'DAY_OF_WEEK', 'MONTH']]

#Confirm no missing values or datatype errors are present
flights_final.isna().sum()

#Create feature set for independent variable
ffx = flights_final[['SCHEDULED_DEPARTURE', 'DISTANCE', 'DAY_OF_WEEK', 'MONTH']]

#Create output set for dependent variable
ffy = flights_final['ARRIVAL_DELAY']

#Create new DataFrame for dependent variable
ffy = pd.DataFrame(ffy)

#Classify flights as late if their arrival delay time is > 0 minutes (Value = 1) otherwise if on-time or early (Value = 0)
ffy['IS_LATE'] = (ffy['ARRIVAL_DELAY'] > 0).astype(int)

#Check head of file to confirm flights with negative arrival delays are classified as 0 and flights with positive values are classified as 1
ffy.head()

#Drop "Arrival Delay" column from analysis
ffy = ffy['IS_LATE']

#Split the data into training and testing sets to validate approach
X_train, X_test, y_train, y_test = train_test_split(ffx, ffy, test_size=0.2, random_state=42)

#Initialize and train an XGBoost classifier
model = XGBClassifier()
model.fit(X_train, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the model accuracy
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

#Display model accuracy and classification report results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report_result)

#Show feature based on features to determine which features impact prediction results
feature_importance = pd.DataFrame({'Feature': ffx.columns, 'Importance': model.feature_importances_})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))
```


#### 4.Test & Validated Model Results
<img src="images/Classification Report.png" alt="Tableau Desktop">


#### 5.Connected in Tableau and Created Dashboards to Support Model 

<img src="images/Connecting2 Data.png" alt="Tableau Desktop">

<img src="images/Dashboard 1.png" alt="Tableau Desktop">

<img src="images/Dashboard 2.png" alt="Tableau Desktop">


## Results
The output of our classification model indicated that it is able to correctly predict flight delays 65% of the time.
- The time at which a flight is scheduled to depart is the most influential factor in predicting flight delays.
- If the scheduled departure time is later in the day or during peak hours, it could contribute to a higher likelihood of delays.
- The time at which a flight is scheduled to depart is the most influential factor in predicting flight delays.
- If the scheduled departure time is later in the day or during peak hours, it could contribute to a higher likelihood of delays.
- The day of the week when a flight is scheduled plays a role in predicting delays.
- Flights scheduled on certain days, possibly weekends or weekdays, might be more prone to delays. This could be influenced by factors like increased air traffic on specific days or different operational schedules.
- The distance between the departure and arrival locations also contributes to predicting flight delays.
- Longer flights might have different operational considerations or potential for delays compared to shorter flights. Factors like layovers or different airport conditions could influence this.

  Understanding these factors can help in making informed decisions, such as avoiding peak departure times or being cautious during certain months, to minimize the risk of flight delays.

## Challenges 

The initial approach was to focus on gathering historical data using an API, however, the free options available severely limited daily endpoint pulls with historical data typically requiring paid versions. Once the historical data was obtained from Kaggle, the next challenge was to one-hot encode the categorical variables for classification analysis. It was quickly identified that the dataset was too large to efficiently encode/explore as it created approximately 1300 columns of airline, destination and arrival airpots, so a test set of 10% was created to solve this challenge and expedite the process. 

With the test dataset created the next challenge was identifying features that were relevant. A correlation matrix and analysis suggested removing a number of variables, and the first iteration of model classification yielded that the one-hot encoded variables were not adding significant prediction accuracy to the model. This allowed for simplification to a number of core features. 

The final challenge was optimizing accuracy by tuning various hyperparameters such as learning rate, tree depth, and iteration numbers to maximize accuracy. This was challenging as it was a new concept but it was overcome through reading and understanding the core value of each hyperparameter.

## Future Goals

Overall, the main features resulted in a prediction accuracy of 65%, which leaves room for overall improvement. Given additional time and resources, future goals would include gathering comparable data across multiple years to avoid any biases towards industry trends that could have affected prediction outcome. Furthermore, gathering additional feature variables for analysis such as weather patterns would have added an additional level of analysis to the overall classification model. 
