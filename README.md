# Titanic-data
Data Cleaning and Preprocessing
1. **Import Libraries**
   - I imported tools (libraries) to handle data (pandas, numpy), scale numbers (StandardScaler from sklearn), and make graphs (matplotlib, seaborn).

2. **Load the Dataset**
   - I downloaded the Titanic data from a website using pandas and put it into a table (dataframe) called `df`.

3. **Preview and Check Data**
   - I looked at the first 5 rows to see what my data looks like.
   - I checked what kind of data is in each column and whether anything is missing.

4. **Handle Missing Values**
   - If any ages were missing, I replaced them with the middle value (median) of all ages.
   - If any fares were missing, I filled them with the average (mean) fare.

5. **Encode Categorical Data**
   - I changed the 'Sex' column so that 'male' becomes 0 and 'female' becomes 1. This turns words into numbers for the machine.

6. **Standardize Numeric Data**
   - I used StandardScaler to make the 'Age' and 'Fare' columns have a similar scale, so they are easier to compare and work better for machine learning.

7. **Visualize Outliers**
   - I drew “boxplots” to see if any ages or fares are much higher or lower than most others (outliers).

8. **Remove Outliers**
   - I calculated the range where most normal values lie (using something called IQR).
   - I removed rows with very unusual 'Fare' or 'Age' values that might mess up your analysis.

9. **Review the Cleaned Data**
   - I printed info and the first few rows again to check how your data looks after cleaning.

**In summary:**
I loaded Titanic data, fixed missing values, changed text to numbers, scaled the numbers, spotted and removed unusual data, and made sure the table is now ready for smart data analysis or machine learning.
