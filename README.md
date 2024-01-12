# LHL-Midterm-Project


## Project/Goals
The goal of this project was to analyze a flight data set that housed information including, but not limited to measures like arrival and dearture times, departure and arrival delays, distance, airline, destination and origin airport, and predict which would be contributing factors to a flight delay. We took advantage of using a classification model with xgboost to try and accurately predict this. 


## Process

#### 1.Identified Target Dataset


#### 2.Conduct EDA
##### Cleaned data, Connected in Tableau and One-hot encoded target variables
``` python
#Check which columns have too many rows of empty data

flights.isna().sum()
```
<img src="images/EDA.png" alt="Tableau Desktop">

``` python
#Based on analysis above, we can remove the columns below as they are mostly blank and will delete the entire dataset of we clean all columns without data. Also removing columns such as "Airline", "Origin Airport", "Destination Airport", and "Tail Number" such those aren't numerical and cannot be part of EDA

columns_to_drop = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'TAIL_NUMBER', 'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DEPARTURE_TIME']
flights.drop(columns = columns_to_drop, inplace = True)
```




<img src="images/Connecting Data.png" alt="Tableau Desktop">

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


#### 5.Created Tabeleau Dashboard to Support Model


## Results

## Challenges 



## Future Goals
