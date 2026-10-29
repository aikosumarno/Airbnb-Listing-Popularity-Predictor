# Airbnb Listing Popularity Predictor

### What is Airbnb?
Airbnb is an online platform that enables property owners to rent out their accommodations to guests for both short-term and long-term stays. It is widely used by travellers around the world who are looking for places to stay during their trips. From private bedrooms, apartments, vacation homes, boutique hotels, cottages and more, Airbnb allows guests to search for their right accommodation based on their desired location, with the features they want and within their price range easily. 

For hosts, especially homeowners, the platform makes it easy to list their properties and earn money from spaces that might otherwise go unused. In addition to photos, descriptions and host information such as Superhost status, Airbnb also allows both guests and hosts to write reviews and rate each other too. This creates a sense of mutual credibility and acts as a form of verification. Reviews help future guests understand what the property is actually like beyond what the host has listed, giving them real insight from previous visitors. 

Since more popular listings typically accumulate a higher number of reviews, this creates an opportunity to develop a predictive model that estimates the potential popularity of new listings before they are posted. Hence, this could potentially help hosts create more appealing listings from the start.

### Problem Definition and Data Understanding
For my HW5 project, I used the Kaggle [dataset](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data) of New York City Airbnb listings from 2019. This data includes information about each listing, the host, geographical attributes, and various other metrics related to availability and performance. The features included in the dataset, along with their descriptions, are as follows:  
- **id**: A unique identifier for the listing.
- **name**: A descriptive name of the listing. 
- **host_id**: A unique identifier for the host.
- **host_name**: The name of the host.
- **neighbourhood_group**: The broader districts within New York City (Brooklyn, Manhattan, Queens, Bronx, Staten Island) 
- **neighbourhood**: The specific neighbourhood where the listing is located (for example, Williamsburg, Bedford-Stuyvesant, Harlem, Bushwick, Upper West Side).
- **latitude**: The latitude coordinate of the property.  
- **longitude**: The longitude coordinate of the property.
- **room_type**: The type of space being listed, such as the entire home or apartment, a private room or a shared room. 
- **price**: The price per night in dollars. 
- **minimum_nights**: The minimum number of nights required for a booking. 
- **number_of_reviews**: The total number of reviews the listing has received. 
- **last_review**: The date of the most recent review.
- **reviews_per_month**: The average number of reviews the listing receives per month. 
- **calculated_host_listings_count:** The total number of listings owned by the host. 
- **availability_365**: The number of days in a year that the listing is available for booking. 
In this project, I will be predicting the variable `reviews_per_month` as a proxy for the popularity of the listing. A model like this could help Airbnb estimate the potential popularity of future listings they are planning to post, which may also guide hosts in creating more appealing and competitive listings.

### Exploratory Data Analysis
Before building my model, I conducted an exploratory data analysis to better understand the dataset and the characteristics of each variable. From this analysis, I observed that name, host_name, neighborhood_group, neighborhood, and room_type are categorical features, while last_review is a date feature. The columns latitude, longitude, price, minimum_nights, number_of_reviews, calculated_host_listings_count, and availability_365 are numerical features. The target variable for this project is reviews_per_month, and the columns id and host_id are simply identifiers and can be safely dropped. 

The values for reviews_per_month range from 0.01 to 58.50. A value of 0.01 is effectively equivalent to zero reviews, since reviews cannot be a decimal number. During this analysis, I also found that several features contained missing or null entries stored as "NaN". As a result, data imputation will be required to ensure that these missing values are handled appropriately. Entries with a missing value for reviews_per_month should also be dropped because data imputation would simply be the same as the purpose of the model that we are building. 

I also observed a strong relationship between price and reviews_per_month. Lower-priced listings generally receive more reviews per month. The distribution of price is also heavily right-skewed, with many listings clustered at lower price points and fewer listings at higher prices. 

A similar trend also appears for minimum_nights. Listings with lower minimum stay requirements tend to accumulate more reviews per month. This feature is also right-skewed, with most hosts setting relatively low

### Feature Engineering
Based on preliminary research on Airbnb and how guests choose their accommodations, I found that proximity to the city center is an important factor in the decision-making process. Because of this, I decided to use the latitude and longitude feature listing's location to compute its distance from the center of New York City. Travellers often consider how close a property is to the city center, as proximity generally influences convenience and accessibility. Therefore, creating a new feature that measures the distance of each listing from the city center could add valuable predictive power to the model. 

The coordinates of central Manhattan are approximately longitude -73.9794 and latitude 40.7639. Since one degree of latitude or longitude corresponds to roughly 111km, I used the Euclidean distance formula with a conversion factor of 111km per degree to calculate the distance of each listing from the city center. 

Using this formula, along with the code shown in Figure 3 below, I computed the distance from each listing to the center of New York City. 

### Preprocessing and Transformations
As mentioned previously, some instances in the dataset contain missing target values. Since models cannot be trained on unkown target variables, these rows must be removed. For simplicity, I also dropped rows with missing values in the reviews_per_month feature. 

Next I applied one-hot encoding to the categorical variables so they could be converted into a numerical format that the model can interpret. For the numerical features, I used a Simple Imputer with the mean strategy to fill in any of the remaining missing values. Then, I applied a StandardScaler to normalize all numerical features, ensuring that they are on a comparable scales and weighted fairly during model training. 

Finally, I dropped the features id, host_id, host_name, latitude, longitude, last_review, calculated_host_listings_count, and name. As mentioned earlier, id and host_id can be safely removed because they are simply unique identifiers and do not contribute meaningful information to the model. Similarly, host_name and name are also unique to each listing or host and do not provide useful predictive value. The latitude and longitude columns have already been used to create the engineered feature distance_to_center, so they are no longer necessary. The feature calculated_host_listings_count showed no clear relationship with reviews_per_month during exploratory analysis, so it was removed as well. Lastly, the name feature was dropped because it is unlikely to have strong predictive power and would require additional text preprocessing steps that are not essential for this project.

### Model Creation 
Once all the features were created and preprocessed, I began model creation and testing.

The list of models I explored is as follows:
- Dummy Regressor (baseline model)
- Linear Regression with Grid Search and Randomized Search for hyperparameter optimization
- RandomForestRegressor
- GradientBoostingRegressor
- RGBRegressor

### Interpretation of Results

Since this is a regression problem, suitable evaluation metrics include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R^2. Because the goal is to predict the popularity of a listing using the target feature reviews_per_month, RMSE is the most appropriate primary metric. It is expressed in the same units as the target variable and penalizes large errors more heavily, making it useful for capturing significant deviations. A lower RMSE indicates better performance, while a higher RMSE reflects poorer predictive accuracy.

Based on the results, the best-performing model for the New York City Airbnb dataset is XGBoost, with a test RMSE of 1.124 and a test R^2 of 0.0526. Random Forest and Gradient Boosting performed similarly, with test RMSE values around 1.17.

### Caveats of the model 
1. reviews_per_month is an imperfect measure of popularity.
Not all guests leave reviews unless they had a very positive or very negative experience. Review frequency can also vary based on guest demographics or the type of listing. As a result, two equally popular listings may have very different review rates, making this an imperfect representation for true demand.

2. Imputation and data removal may introduce bias.
Missing values were imputed using the mean under the assumption that it reasonably represented the data. However, this may oversimplify the true distribution and reduce variation. Additionally, rows with missing target values had to be removed, which may have led to an uneven data distribution or subtle bias in the training set.

3. Some dropped features may have contained valuable information.
For simplicity, several features were removed, but many of them might still hold predictive value when used correctly. For example, a descriptive listing title could indirectly reflect amenities, style, or uniqueness, all of which might influence guest decisions. Removing these features simplifies processing but may limit predictive performance.

### Key Takeaways 
Throughout this assignment, I learned how essential proper feature preprocessing is for building effective machine learning models. Without careful preprocessing, issues can arise during feature selection, scaling, and hyperparameter optimization, ultimately affecting model performance.

Exploring different models showed that while ensemble tree-based methods are more complex, they tend to perform better overall. The accuracy gains were significant enough to justify the added complexity. However, even after hyperparameter tuning and feature reduction, the improvements across models remained relatively small, suggesting that the current feature set and model choices already capture most of the available predictive signal in the dataset.

For future work, I hope to conduct more extensive hyperparameter optimization for XGBoost and incorporate interpretability tools such as SHAP and ELI5 to better understand feature contributions and improve model transparency.
