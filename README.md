# **Predict students' dropout and academic success**

-- ELEC3544 Group Project

---
## --------------------- Workflow ---------------------
### **Phase 1: Data Acquisition & Initial Scoping**
* **Data Ingestion:** Load the dataset and verify dimensions.
* **Target Definition:** Identify the 3-class target (`Dropout`, `Enrolled`, `Graduate`).
* **Mapping:** Convert categorical IDs into readable strings for analysis (if necessary) to ensure the EDA makes sense to a human reader.
### **Phase 2: Systematic Data Quality Assessment (Cleaning)**
* **Structure:** Check data types (Integer vs. Float). Ensure categorical variables are handled as such and not treated as continuous numbers.
* **Completeness:** Check for missing values (The UCI dataset is famously "clean," but you must demonstrate that you checked).
* **Validity:**
  * Verify numerical ranges (e.g., Grades should be between 0–20).
  * Check for logical errors (e.g., Are there students with "0 units enrolled" but a "grade of 15"?).
* **Distributional:** Check for skewness in features like `Age at enrollment`. Decide if outliers (mature students) should be capped or kept.
### **Phase 3: Exploratory Data Analysis (EDA)**
* **Correlation Block Analysis:**
  1. Academic Block: Identifying redundancy between 1st and 2nd semesters.
  2. Socio-Economic Block: Assessing parental influence.
  3. Financial Block: Testing the impact of debt/scholarships.
  4. Demographic Block: Checking for bias (`Gender` / `Age`).
* **Feature-Target Relationship:** Visualizing how the top correlated features (like *Grades* or *Tuition* status) differ across the three Target classes using Boxplots or Violin plots.
### **Phase 4: Feature Selection & Engineering**
* **Dimensionality Reduction:**
  * Drop redundant features (e.g., dropping `Nationality` if `International` provides the same info).
  * Drop low-variance features (e.g., `GDP`, `Inflation`, and `Unemployment` if they show near-zero correlation with the target).
* **Feature Construction:** (Optional) Create an "Academic Progression" feature by calculating the difference between 2nd-semester grades and 1st-semester grades.
* **Data Transformation:** Apply StandardScaler to numerical features (essential for Logistic Regression) and One-Hot Encoding for categorical features that aren't ordinal.
### **Phase 5: Model Selection & Training Strategy**
* **Baseline Model (Logistic Regression):**
  * **Goal:** Establish a "transparent" benchmark.
  * **Action:** Analyze coefficients to explain which factors increase the probability of dropout.
* **Advanced Model (Random Forest):**
  * **Goal:** Capture non-linear relationships and interactions between variables.
  * **Action:** Compare performance against the baseline.
* **Cross-Validation:** Use 5-fold Stratified Cross-Validation to ensure the model isn't just "getting lucky" on a specific slice of data.
### **Phase 6: Model Evaluation & Interpretation**
* **Metric Selection:**
  * Don't just use **Accuracy**.
  * Use a **Confusion Matrix** to see if the model confuses "Enrolled" with "Dropout."
  * Use **F1-Score** and **Recall** (Recall is critical here because missing a potential dropout is more "expensive" for a university than misidentifying a graduate).
* **Feature Importance:** Compare the "Weights" from Logistic Regression with the "Importance" from Random Forest.
### **Phase 7: Post-Mortem & Recommendations - [Coursework Conclusion]**
* **The "So What?":** Translate the math into policy. (e.g., "Since 'Tuition fees up to date' is the strongest predictor, the university should implement financial counseling for students in debt by the end of Semester 1.")
* **Limitations:** Discuss what data was missing (e.g., Mental health data or Social integration data) that could have made the model better.
* **Final Summary:** Conclude whether the project succeeded in creating an **"Early Warning System"**.

# **Step 1: Setup and Loading Data**


```python
# Install the UCI Repository library
!pip install ucimlrepo -q
```


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# Data (as pandas dataframes)
df = predict_students_dropout_and_academic_success.data.features
df = df.rename(columns={'Nacionality': 'Nationality'})
target = predict_students_dropout_and_academic_success.data.targets

# Combine them for a full quality assessment
full_df = pd.concat([df, target], axis=1)

# 數值化 Target (Dropout=0, Enrolled=1, Graduate=2)
target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
full_df['Target_num'] = full_df['Target'].map(target_map)

print("Dataset loaded successfully!")
full_df.head()
```

    Dataset loaded successfully!






  <div id="df-b9642354-6f8b-44fd-8ad3-412d562a8fd2" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Marital Status</th>
      <th>Application mode</th>
      <th>Application order</th>
      <th>Course</th>
      <th>Daytime/evening attendance</th>
      <th>Previous qualification</th>
      <th>Previous qualification (grade)</th>
      <th>Nationality</th>
      <th>Mother's qualification</th>
      <th>Father's qualification</th>
      <th>...</th>
      <th>Curricular units 2nd sem (enrolled)</th>
      <th>Curricular units 2nd sem (evaluations)</th>
      <th>Curricular units 2nd sem (approved)</th>
      <th>Curricular units 2nd sem (grade)</th>
      <th>Curricular units 2nd sem (without evaluations)</th>
      <th>Unemployment rate</th>
      <th>Inflation rate</th>
      <th>GDP</th>
      <th>Target</th>
      <th>Target_num</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>17</td>
      <td>5</td>
      <td>171</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>19</td>
      <td>12</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15</td>
      <td>1</td>
      <td>9254</td>
      <td>1</td>
      <td>1</td>
      <td>160.0</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.666667</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>9070</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>37</td>
      <td>37</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>10.8</td>
      <td>1.4</td>
      <td>1.74</td>
      <td>Dropout</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>17</td>
      <td>2</td>
      <td>9773</td>
      <td>1</td>
      <td>1</td>
      <td>122.0</td>
      <td>1</td>
      <td>38</td>
      <td>37</td>
      <td>...</td>
      <td>6</td>
      <td>10</td>
      <td>5</td>
      <td>12.400000</td>
      <td>0</td>
      <td>9.4</td>
      <td>-0.8</td>
      <td>-3.12</td>
      <td>Graduate</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>39</td>
      <td>1</td>
      <td>8014</td>
      <td>0</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
      <td>37</td>
      <td>38</td>
      <td>...</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>13.000000</td>
      <td>0</td>
      <td>13.9</td>
      <td>-0.3</td>
      <td>0.79</td>
      <td>Graduate</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-b9642354-6f8b-44fd-8ad3-412d562a8fd2')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-b9642354-6f8b-44fd-8ad3-412d562a8fd2 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-b9642354-6f8b-44fd-8ad3-412d562a8fd2');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




# **Step 2: Systematic data quality assessment**
## 1. Structure Assessment


```python
print("--- STRUCTURE ASSESSMENT ---")
# Check dimensions
print(f"Dataset Shape: {full_df.shape}")

# Check data types and memory usage
print("\nData Types and Info:")
print(full_df.info())

# Check for the column names to ensure they are clean
print("\nColumn Names:")
print(full_df.columns.tolist())
```

    --- STRUCTURE ASSESSMENT ---
    Dataset Shape: (4424, 38)
    
    Data Types and Info:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4424 entries, 0 to 4423
    Data columns (total 38 columns):
     #   Column                                          Non-Null Count  Dtype  
    ---  ------                                          --------------  -----  
     0   Marital Status                                  4424 non-null   int64  
     1   Application mode                                4424 non-null   int64  
     2   Application order                               4424 non-null   int64  
     3   Course                                          4424 non-null   int64  
     4   Daytime/evening attendance                      4424 non-null   int64  
     5   Previous qualification                          4424 non-null   int64  
     6   Previous qualification (grade)                  4424 non-null   float64
     7   Nationality                                     4424 non-null   int64  
     8   Mother's qualification                          4424 non-null   int64  
     9   Father's qualification                          4424 non-null   int64  
     10  Mother's occupation                             4424 non-null   int64  
     11  Father's occupation                             4424 non-null   int64  
     12  Admission grade                                 4424 non-null   float64
     13  Displaced                                       4424 non-null   int64  
     14  Educational special needs                       4424 non-null   int64  
     15  Debtor                                          4424 non-null   int64  
     16  Tuition fees up to date                         4424 non-null   int64  
     17  Gender                                          4424 non-null   int64  
     18  Scholarship holder                              4424 non-null   int64  
     19  Age at enrollment                               4424 non-null   int64  
     20  International                                   4424 non-null   int64  
     21  Curricular units 1st sem (credited)             4424 non-null   int64  
     22  Curricular units 1st sem (enrolled)             4424 non-null   int64  
     23  Curricular units 1st sem (evaluations)          4424 non-null   int64  
     24  Curricular units 1st sem (approved)             4424 non-null   int64  
     25  Curricular units 1st sem (grade)                4424 non-null   float64
     26  Curricular units 1st sem (without evaluations)  4424 non-null   int64  
     27  Curricular units 2nd sem (credited)             4424 non-null   int64  
     28  Curricular units 2nd sem (enrolled)             4424 non-null   int64  
     29  Curricular units 2nd sem (evaluations)          4424 non-null   int64  
     30  Curricular units 2nd sem (approved)             4424 non-null   int64  
     31  Curricular units 2nd sem (grade)                4424 non-null   float64
     32  Curricular units 2nd sem (without evaluations)  4424 non-null   int64  
     33  Unemployment rate                               4424 non-null   float64
     34  Inflation rate                                  4424 non-null   float64
     35  GDP                                             4424 non-null   float64
     36  Target                                          4424 non-null   object 
     37  Target_num                                      4424 non-null   int64  
    dtypes: float64(7), int64(30), object(1)
    memory usage: 1.3+ MB
    None
    
    Column Names:
    ['Marital Status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance', 'Previous qualification', 'Previous qualification (grade)', 'Nationality', "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation", 'Admission grade', 'Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment', 'International', 'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP', 'Target', 'Target_num']


## 2. Completeness Assessment


```python
print("--- COMPLETENESS ASSESSMENT ---")

# Count missing values per column
missing_values = full_df.isnull().sum()
missing_percentage = (missing_values / len(full_df)) * 100

# Display only columns that have missing values (if any)
if missing_values.sum() == 0:
    print("Success: No missing values detected in the dataset.")
else:
    print("Missing values found:")
    print(missing_values[missing_values > 0])

# Visualizing completeness (Useful for large datasets)
plt.figure(figsize=(10, 4))
sns.heatmap(full_df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap (Purple = Data Present, Yellow = Missing)")
plt.show()
```

    --- COMPLETENESS ASSESSMENT ---
    Success: No missing values detected in the dataset.



    
![png](3544_Project_files/3544_Project_7_1.png)
    


## 3. Validity Assessment


```python
print("--- VALIDITY ASSESSMENT ---")

# 1. Check for Duplicate Rows
duplicates = full_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# 2. Logical Range Check (Summary Statistics)
# We look for impossible values like negative 'Age at enrollment' or 'Curricular units 1st sem (grade)' > 20
print("\nDescriptive Statistics for Range Validation:")
display(full_df[['Age at enrollment', 'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']].describe())

# 3. Categorical Validity
# Check the unique values in the Target variable
print("\nUnique values in Target (Should be Dropout, Graduate, or Enrolled):")
print(full_df['Target'].unique())
```

    --- VALIDITY ASSESSMENT ---
    Number of duplicate rows: 0
    
    Descriptive Statistics for Range Validation:




  <div id="df-5adf337f-5e40-415a-89a4-925b5d664e85" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age at enrollment</th>
      <th>Curricular units 1st sem (grade)</th>
      <th>Curricular units 2nd sem (grade)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4424.000000</td>
      <td>4424.000000</td>
      <td>4424.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.265145</td>
      <td>10.640822</td>
      <td>10.230206</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.587816</td>
      <td>4.843663</td>
      <td>5.210808</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>19.000000</td>
      <td>11.000000</td>
      <td>10.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20.000000</td>
      <td>12.285714</td>
      <td>12.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>25.000000</td>
      <td>13.400000</td>
      <td>13.333333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>70.000000</td>
      <td>18.875000</td>
      <td>18.571429</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5adf337f-5e40-415a-89a4-925b5d664e85')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5adf337f-5e40-415a-89a4-925b5d664e85 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5adf337f-5e40-415a-89a4-925b5d664e85');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>



    
    Unique values in Target (Should be Dropout, Graduate, or Enrolled):
    ['Dropout' 'Graduate' 'Enrolled']


## 4. Distributional Assessment


```python
print("--- DISTRIBUTIONAL ASSESSMENT ---")

# 1. Target Variable Distribution (Class Imbalance check)
plt.figure(figsize=(8, 5))
sns.countplot(x='Target',legend=False, data=full_df, hue='Target')
plt.title("Distribution of the Target Variable (Class Balance)")
plt.show()

# 2. Feature Distribution (Checking for Skewness/Outliers)
# Let's look at 'Age at enrollment'
plt.figure(figsize=(10, 5))
sns.histplot(full_df['Age at enrollment'], kde=True, color='blue')
plt.title("Distribution of Age at Enrollment")
plt.show()

# 3. Correlation check (Detecting Multicollinearity)
plt.figure(figsize=(12, 10))
# Correlation only for numerical columns
corr = full_df.select_dtypes(include=['number']).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".1f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

    --- DISTRIBUTIONAL ASSESSMENT ---



    
![png](3544_Project_files/3544_Project_11_1.png)
    



    
![png](3544_Project_files/3544_Project_11_2.png)
    



    
![png](3544_Project_files/3544_Project_11_3.png)
    


# **Part 3: Handling High Correlations**
We use 1) **DROP**, 2) **COMBINE** or 3) **KEEP** (Model Regularzation) to handle high correlations features
## 1. Academic Features


```python
# 1. Academic Features List
academic_cols = [
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)'
]
plt.figure(figsize=(12, 10))
corr_academic = full_df[academic_cols].corr()

sns.heatmap(corr_academic, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation: Academic Performance', fontsize=15, pad=20)
plt.show()

# Insight: Look for values > 0.80. These are candidates for combining into an "Annual Average".
```


    
![png](3544_Project_files/3544_Project_13_0.png)
    


### Feature Engineering (Combining features)
**Feature Engineering Strategy**: Instead of using raw semester-wise data which exhibited high multicollinearity, we synthesized four behavioral metrics: **Grade Trend, Approved Difference, Success Rate, and Average Grade**. This approach reduces the feature space while emphasizing the dynamic academic progression of students, which is hypothesized to be a stronger predictor of dropout than static semester snapshots.


```python
import pandas as pd
import numpy as np

# 1. 創建新的工程特徵 (New Engineered Features)

# A. 成績趨勢：正值代表進步，負值代表退步
full_df['Grade_Trend'] = full_df['Curricular units 2nd sem (grade)'] - full_df['Curricular units 1st sem (grade)']

# B. 學分通過差異：觀察學生是否在第二學期適應得更好
full_df['Approved_Diff'] = full_df['Curricular units 2nd sem (approved)'] - full_df['Curricular units 1st sem (approved)']

# C. 學分成功率 (Success Rate)：總共通過多少 / 總共註冊多少
# 注意：為了避免除以零，加上一個極小值 epsilon
total_enrolled = full_df['Curricular units 1st sem (enrolled)'] + full_df['Curricular units 2nd sem (enrolled)']
total_approved = full_df['Curricular units 1st sem (approved)'] + full_df['Curricular units 2nd sem (approved)']
full_df['Credit_Success_Rate'] = total_approved / (total_enrolled + 1e-5)

# D. 平均學術表現
full_df['Avg_Grade'] = (full_df['Curricular units 1st sem (grade)'] + full_df['Curricular units 2nd sem (grade)']) / 2

# 2. 定義需要移除的原始高相關特徵 (原本那 12 個)
original_academic_cols = [col for col in full_df.columns if 'sem' in col]

# 3. 建立簡化後的 dataframe 用於後續建模
# 我們保留新特徵，移除舊的學術特徵
reduced_df = full_df.drop(columns=academic_cols)

print("New features have been established, and the original academic features have been removed.")
print(f"Number of features remaining: {len(reduced_df.columns)}")
reduced_df[['Grade_Trend', 'Approved_Diff', 'Credit_Success_Rate', 'Avg_Grade']].head()
```

    New features have been established, and the original academic features have been removed.
    Number of features remaining: 30






  <div id="df-de4538f2-3bc2-47fa-b046-bb6c8f746b48" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Grade_Trend</th>
      <th>Approved_Diff</th>
      <th>Credit_Success_Rate</th>
      <th>Avg_Grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.333333</td>
      <td>0</td>
      <td>0.999999</td>
      <td>13.833333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.028571</td>
      <td>-1</td>
      <td>0.916666</td>
      <td>12.914286</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.666667</td>
      <td>1</td>
      <td>0.916666</td>
      <td>12.666667</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-de4538f2-3bc2-47fa-b046-bb6c8f746b48')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-de4538f2-3bc2-47fa-b046-bb6c8f746b48 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-de4538f2-3bc2-47fa-b046-bb6c8f746b48');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>

    </div>
  </div>




**Verify: Correlation between engineered features & target**


```python
# 1. 確保 Target 已數值化 (Dropout=0, Enrolled=1, Graduate=2)
# 如果之前已經做過，這行會直接執行
if 'Target_num' not in full_df.columns:
    target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    full_df['Target_num'] = full_df['Target'].map(target_map)

# 2. 定義你建立的簡化特徵列表
reduced_features = ['Grade_Trend', 'Approved_Diff', 'Credit_Success_Rate', 'Avg_Grade']

# 3. 計算相關性矩陣，並只提取與 Target_num 相關的那一列
correlation_with_target = full_df[reduced_features + ['Target_num']].corr()['Target_num'].drop('Target_num')

# 4. 排序以利觀察
correlation_with_target = correlation_with_target.sort_values(ascending=False)

# 5. 繪製條形圖 (視覺上比熱圖更直觀地對比各特徵的貢獻)
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index, hue=correlation_with_target.index, legend=False)

# 加上數值標籤
for i, v in enumerate(correlation_with_target.values):
    plt.text(v + 0.01, i, f'{v:.2f}', color='black', va='center', fontweight='bold')

plt.title('Correlation between Reduced Features and Target', fontsize=14)
plt.xlabel('Correlation Coefficient')
plt.ylabel('Engineered Features')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1) # 增加中心線
plt.show()
```


    
![png](3544_Project_files/3544_Project_17_0.png)
    


## 2. Socio-Economic Features


```python
# 2. Socio-economic Features
socio_cols = [
    'Marital Status', 'Gender', 'Age at enrollment', 'Displaced',
    'Educational special needs', 'Debtor', 'Tuition fees up to date',
    'Scholarship holder', 'International', 'Nationality'
]

plt.figure(figsize=(10, 8))
corr_socio = full_df[socio_cols].corr()
mask = np.triu(np.ones_like(corr_socio, dtype=bool))

sns.heatmap(corr_socio, annot=True, fmt=".2f", cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Feature Correlation: Socio-economic Factors', fontsize=15, pad=20)
plt.show()

# Insight: Often Mother's and Father's qualifications are correlated.
# You might simplify this later to "Highest Parental Education Level".
```


    
![png](3544_Project_files/3544_Project_19_0.png)
    


We shall remove **Nationality** as it is redundant with "International"


```python
if "Nationality" in reduced_df.columns:
    reduced_df = reduced_df.drop(columns=["Nationality"])


print("Removed Nationality")
print(f"- Number of features remaining: {len(reduced_df.columns)}")
```

    Removed Nationality
    - Number of features remaining: 29


## 3. Enrollment Features


```python
# 3. Enrollment Features List
background_cols = [
    'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Admission grade',
    'Mother\'s qualification', 'Father\'s qualification',
    'Mother\'s occupation', 'Father\'s occupation'
]

plt.figure(figsize=(12, 10))
corr_back = full_df[background_cols].corr()
mask = np.triu(np.ones_like(corr_back, dtype=bool))

sns.heatmap(corr_back, annot=True, fmt=".2f", cmap='BrBG', center=0, linewidths=0.5)
plt.title('Feature Correlation: Enrollment Background', fontsize=15, pad=20)
plt.show()

# Insight: 'Debtor' and 'Tuition fees up to date' usually have a strong negative correlation.
# If a student is a debtor, their fees are likely not up to date.
```


    
![png](3544_Project_files/3544_Project_23_0.png)
    


We shall remove **Father's occupation** as it indicating extreme redundancy


```python
# 1. 處理父母職業的高相關性 (0.91)
# 由於兩者極度冗餘，我們移除 Father's occupation
if "Father's occupation" in reduced_df.columns:
    reduced_df = reduced_df.drop(columns=["Father's occupation"])

# 2. 關於父母學歷 (0.54) 和成績 (0.58)，我們選擇保留
# 因為它們保留了足夠的獨立資訊 (Unique Information)

print("- Removed Father's occupation (r=0.91)。")
print(f"- Number of features remaining: {len(reduced_df.columns)}")
```

    - Removed Father's occupation (r=0.91)。
    - Number of features remaining: 28


## 4. Macroeconomic Features


```python
# 4. Macroeconomic Features List
macro_cols = ['Unemployment rate', 'Inflation rate', 'GDP']

plt.figure(figsize=(8, 6))
corr_macro = full_df[macro_cols].corr()
mask = np.triu(np.ones_like(corr_macro, dtype=bool))

sns.heatmap(corr_macro, annot=True, fmt=".2f", cmap='YlGnBu', center=0, linewidths=0.5)
plt.title('Feature Correlation: Macroeconomic Indicators', fontsize=15, pad=20)
plt.show()

# Insight: 'Nacionality' and 'International' will have near-perfect correlation.
# You should definitely drop one of these two to avoid redundancy.
```


    
![png](3544_Project_files/3544_Project_27_0.png)
    


### **GDP, Unemployment rate, and Inflation rate**

While they are included in the original UCI dataset to provide "macro-economic context," they are frequently dropped or ignored for four specific reasons:

1. The "Granularity Mismatch"
GDP is a macro-level variable (national level), while dropout is a micro-level event (individual level). The Problem: In this dataset, every student who enrolled in the same year is assigned the exact same GDP and Unemployment value. The Result: The model struggles to learn from this because the feature doesn't vary between the "Success" student and the "Dropout" student who started in the same semester. It lacks "discriminatory power."

2. Lack of Variance
Because the data was collected over a limited number of years, there are only a handful of unique values for GDP in the entire column of thousands of rows.If a feature is almost a constant (or has very few levels), it provides almost zero information to a machine learning model.

3. Indirect vs. Direct Influence (Proxies)
Macro-economic factors like GDP do affect students, but they do so indirectly.Example: A low GDP might lead to a student's father losing his job, which makes the student a "Debtor." The Dataset Logic: Since the dataset already contains direct variables like "Debtor," "Scholarship holder," and "Tuition fees up to date," the "GDP" variable becomes redundant. The direct variables "absorb" the impact of the macro-economic ones.
4. Multicollinearity with other Macro-features
GDP, Unemployment, and Inflation are usually highly correlated with each other (as one goes down, the others move in predictable ways). Including all three can cause mathematical instability in models like Logistic Regression.


```python
# 1. 篩選宏觀經濟特徵
macro_cols = ['Tuition fees up to date', 'Unemployment rate', 'Inflation rate', 'GDP', 'Target_num']
macro_corr = full_df[macro_cols].corr()['Target_num'].drop('Target_num')

# 2. 繪圖
plt.figure(figsize=(8, 5))
sns.barplot(x=macro_corr.values, y=macro_corr.index, palette='magma')
plt.axvline(x=0.1, color='red', linestyle='--', label='Significance Threshold (0.1)')
plt.axvline(x=-0.1, color='red', linestyle='--')
plt.title('Macroeconomic Impact on Target', fontsize=12)
plt.xlabel('Correlation Coefficient')
plt.xlim(-1, 1) # 統一量程以顯示其微小
plt.legend()
plt.show()

print("Macroeconomic features show a low correlation with target (Lower than threshold, 0.1)")
```

    /tmp/ipykernel_1119/3853860225.py:7: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      sns.barplot(x=macro_corr.values, y=macro_corr.index, palette='magma')



    
![png](3544_Project_files/3544_Project_29_1.png)
    


    Macroeconomic features show a low correlation with target (Lower than threshold, 0.1)


*   **Tuition fees up to date:** Usually has a high correlation (approx **0.40+**) with success.
* **GDP:** Usually has a very near-zero correlation (approx **0.04**) with success.

**Verdict for your project:**
* If you are using **Linear Models** (Logistic Regression, SVM), you should **drop** GDP/Unemployment/Inflation to keep the model simple.
* If you are using **Tree-based Models** (Random Forest, XGBoost), you can **keep** them; the model will simply ignore them or give them a "Feature Importance" score of nearly zero, so they won't do much harm, but they won't help much either.


```python
# 移除所有宏觀經濟特徵
macro_to_drop = ['Unemployment rate', 'Inflation rate', 'GDP']
reduced_df = reduced_df.drop(columns=[col for col in macro_to_drop if col in reduced_df.columns])

print(f"Macroeconomic features (3) are removed. Remaining: {len(reduced_df.columns)}")
```

    Macroeconomic features (3) are removed. Remaining: 25


### **Preparing for modeling - Clean Target**


```python
# 1. 移除原始的字串 Target 欄位
if 'Target' in reduced_df.columns:
    reduced_df = reduced_df.drop(columns=['Target'])

# 2. 將 Target_num 重新命名回 Target
if 'Target_num' in reduced_df.columns:
    reduced_df = reduced_df.rename(columns={'Target_num': 'Target'})

print(f"Final Number of features remaining: {len(reduced_df.columns)}")
print(reduced_df.info())
```

    Final Number of features remaining: 24
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4424 entries, 0 to 4423
    Data columns (total 24 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   Marital Status                  4424 non-null   int64  
     1   Application mode                4424 non-null   int64  
     2   Application order               4424 non-null   int64  
     3   Course                          4424 non-null   int64  
     4   Daytime/evening attendance      4424 non-null   int64  
     5   Previous qualification          4424 non-null   int64  
     6   Previous qualification (grade)  4424 non-null   float64
     7   Mother's qualification          4424 non-null   int64  
     8   Father's qualification          4424 non-null   int64  
     9   Mother's occupation             4424 non-null   int64  
     10  Admission grade                 4424 non-null   float64
     11  Displaced                       4424 non-null   int64  
     12  Educational special needs       4424 non-null   int64  
     13  Debtor                          4424 non-null   int64  
     14  Tuition fees up to date         4424 non-null   int64  
     15  Gender                          4424 non-null   int64  
     16  Scholarship holder              4424 non-null   int64  
     17  Age at enrollment               4424 non-null   int64  
     18  International                   4424 non-null   int64  
     19  Target                          4424 non-null   int64  
     20  Grade_Trend                     4424 non-null   float64
     21  Approved_Diff                   4424 non-null   int64  
     22  Credit_Success_Rate             4424 non-null   float64
     23  Avg_Grade                       4424 non-null   float64
    dtypes: float64(5), int64(19)
    memory usage: 829.6 KB
    None


### **Categorical Encoding: One-Hot Encoding**


```python
# 定義需要進行 One-Hot Encoding 的類別特徵
categorical_features = [
    'Marital Status', 'Application mode', 'Course',
    'Previous qualification', "Mother's qualification",
    "Father's qualification", "Mother's occupation"
]

# 使用 pandas 的 get_dummies 進行轉換
df_final = pd.get_dummies(reduced_df, columns=categorical_features, drop_first=True)

print(f"Dimention after One-Hot Encoding: {df_final.shape}")
```

    Dimention after One-Hot Encoding: (4424, 163)


### **Feature Scaling: Standardization**


```python
from sklearn.preprocessing import StandardScaler

# 排除 Target 欄位
X = df_final.drop(columns=['Target'])
y = df_final['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 轉回 DataFrame 方便觀察
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
```

# **Part 4: Choosing a regression model**
![image.png](data:image/png;base64,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G4PaQfHCMk799T7efnLEHpXDDYSoutdD+4oW/lDrTp5XUNyAYvgKl1DBNg61FqSZYkKIBBK/P1tSKJ6YeHh9pI33qsb7alckM58R/1rJvt+VYJc86tUn4XRE17xPsfjsaWc7qCpYxty7jQIxOFlf4JQCsfjsHfcZfXtWlhHAxcZvTKBADJDhpPq3ldj2FFPcMv6+T+KzoBJ6sZiKcB4tjVIgFuMg8VvcLAZidGwqxjujkrZnpaBx+awUNLaQd7aGwh+MMp2fuDwSTkclheLJ4NJFIj1Rc8Q8O0WN2ymaL+/cxQ2gL4RvdwehjGwqf9tIwfTUDcZhVgltXdVYQqE0UxaIG8ZAmK53XbR01+GW5EeMl8RvcTOIdfP3Wjn7fFK7ecUsbuq6W8g/c5DwOGehEeMk4XRUN2/z6QGzj64KynQ7jkzJZY0o3kdZM/s3y6USQgghhBBCCLipC8VCCCGEEEIIIcQVkubTQgghhBBCCCEmLSkUCyGEEEIIIYSYtKRQLIQQQgghhBBi0pJCsRBCCCGEEEKISUsKxUIIIYQQQgghJi0pFAshhBD///buP7it8t7z+PuOuyuGzqrD7ipDZ6OhStyUxFASAXHF0MSZ3MQZLnZK4+pCtE4HrcMgSBsXT0HX29b4NugaUpd06+BtIrw38XoA1W2ulW0m4jZrpcOgCRCZKQjuDUqGoMyFibpkol0Yzm48Z/84x7YsO7acX03Q5zXjyeQ5j/QcPdI55/me58cRERGRiqWgWERERERERCqWgmIRERERERGpWAqKRUREREREpGIpKBYREREREZGKpaBYREREREREKpaCYhEREREREalYCopFRERERESkYikoFhERERERkYqloFhEREREREQqloJiERERERERqVgKikVERERERKRiKSgWERERERGRiqWgWERERERERCqWgmIRERERERGpWAqKRUREREREpGIpKBYREREREZGKpaBYREREREREKpaCYhEREREREalYCopFRERERESkYikoFhERERERkYqloFhEREREREQqloJiERERERERqVgKikVERERERKRiKSgWERERERGRiqWgWERERERERCqWgmIRERERERGpWAqKRUREREREpGIpKBYREREREZGKpaD4WlXIEO/byc7efhLHjdKtIiIiIiIym9ECyW1NRN4o3SDXhNEUEX+E5JnS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)

**Train-Test Split**


```python
from sklearn.model_selection import train_test_split

# Mapping: 0 for Dropout, 1 for Success (combining Enrolled and Graduate)
y_binary = y.map({0: 0, 1: 1, 2: 1})

X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X_scaled_df, y_binary, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}")
print(f"Test: {len(X_test)}")
```

    Train: 3539
    Test: 885



```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 建立一個評估 FUNCTION
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    # 定義對應的名稱
    target_names = ['Dropout', 'Success']

    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)

    # 使用 target_names 作為軸標籤
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14)
    plt.show()
```

## **Baseline (Logistic Regression)**


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. 初始化模型
# max_iter 設為 1000 確保梯度下降能收斂
# Using 'liblinear' for small-to-medium datasets; 'max_iter' increased for convergence
# 使用 'balanced' 讓模型自動根據樣本比例分配權重
log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

# 2. 訓練模型
log_model.fit(X_train, y_train_binary)

# 3. 使用我們之前的評估函數
evaluate_model(log_model, X_test, y_test_binary, model_name="Logistic Regression")
```

    --- Logistic Regression Evaluation ---
    Accuracy: 0.8452
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Dropout       0.74      0.81      0.77       284
         Success       0.90      0.86      0.88       601
    
        accuracy                           0.85       885
       macro avg       0.82      0.83      0.83       885
    weighted avg       0.85      0.85      0.85       885
    



    
![png](3544_Project_files/3544_Project_43_1.png)
    



```python
# 提取對預測 "Dropout" (類別 0) 貢獻最大的特徵
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': log_model.coef_[0]  # 指向 Dropout 類別
})

# 1. 建立對應字典 (根據官方 Metadata)
mapping_dict = {
    # Core Academic & Financial Features
    'Tuition fees up to date': 'Financial: Tuition Paid up to Date',
    'Debtor': 'Financial: Debt Owed (Debtor)',
    'Scholarship holder': 'Financial: Scholarship Holder',

    # Course / Major Information
    'Course_171': 'Course 171: Animation & Multimedia Design',
    'Course_9853': 'Course 9853: Social Service',
    'Course_9130': 'Course 9130: Equiniculture (Horse Management)',
    'Course_9773': 'Course 9773: Journalism and Communication',
    'Course_9991': 'Course 9991: Management (Evening)',
    'Course_9500': 'Course 9500: Nursing',

    # Demographic & Application Mode
    'Age at enrollment': 'Demographic: Age at Enrollment',
    'Application mode_39': 'Application Mode: Mature Student (Over 23 Entry)',
    'Application mode_53': 'Application Mode: Short Cycle Diploma Holders',
    'Previous qualification_4': 'Prev Qual: High School Incomplete',

    # Maternal Background (Occupation - Crucial for your LogReg results)
    'Mother\'s occupation_9': 'Mother\'s Job: Scientific/Intellectual Profession',
    'Mother\'s occupation_4': 'Mother\'s Job: Administrative Staff',
    'Mother\'s occupation_134': 'Mother\'s Job: Skilled Worker (Industry)',
    'Mother\'s occupation_144': 'Mother\'s Job: Manual Worker (Agriculture/Fishery)',
    'Mother\'s occupation_191': 'Mother\'s Job: Unskilled Worker (Services)',
    'Mother\'s occupation_90': 'Mother\'s Job: Other Unskilled Worker',

    # Parental Education (Qualification)
    'Mother\'s qualification_11': 'Mother\'s Edu: Primary Education (1st Cycle)',
    'Mother\'s qualification_34': 'Mother\'s Edu: Higher Education (Degree)',
    'Father\'s qualification_34': 'Father\'s Edu: Unknown', # Based on common UCI encoding 34 is often Higher Ed or Unknown; please verify with your metadata
}

# 2. 處理你的特徵係數結果
# 假設你的特徵係數儲存在 feature_importance 這個 DataFrame 中
feature_importance['Factor'] = feature_importance['Feature'].map(mapping_dict).fillna(feature_importance['Feature'])

# 3. 重新排序並顯示
top_10_mapped = feature_importance.sort_values(by='Coefficient', ascending=False).head(10)

print("--- Top 10 Factors Leading to Dropout ---")
print(top_10_mapped[['Factor', 'Coefficient']])
```

    --- Top 10 Factors Leading to Dropout ---
                                                 Factor  Coefficient
    14                              Credit_Success_Rate     1.975923
    7                Financial: Tuition Paid up to Date     0.767217
    158          Mom's Job: Unskilled Worker (Services)     0.610747
    38        Course 171: Animation & Multimedia Design     0.550251
    139   Mom's Job: Scientific/Intellectual Profession     0.398394
    134                 Mom's Job: Administrative Staff     0.316369
    36    Application Mode: Short Cycle Diploma Holders     0.278343
    13                                    Approved_Diff     0.277627
    151  Mom's Job: Manual Worker (Agriculture/Fishery)     0.274619
    148            Mom's Job: Skilled Worker (Industry)     0.263262


## **Random Forest**


```python
from sklearn.ensemble import RandomForestClassifier

# 1. 初始化模型
# n_estimators=100 代表建立 100 棵樹
# random_state=42 確保每次跑的結果都一樣，方便寫報告
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=10, random_state=42)

# 2. 訓練模型
rf_model.fit(X_train, y_train_binary)

# 3. 使用我們之前的評估函數
evaluate_model(rf_model, X_test, y_test_binary, model_name="Random Forest")
```

    --- Random Forest Evaluation ---
    Accuracy: 0.8588
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Dropout       0.76      0.82      0.79       284
         Success       0.91      0.88      0.89       601
    
        accuracy                           0.86       885
       macro avg       0.84      0.85      0.84       885
    weighted avg       0.86      0.86      0.86       885
    



    
![png](3544_Project_files/3544_Project_46_1.png)
    



```python
# 提取特徵重要性
rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
})

# 排序並取前 10 名
top_10_rf = rf_importance.sort_values(by='Importance', ascending=False).head(10)

# 加入我們之前的 Mapping 描述，讓結果更好讀
top_10_rf['Factor'] = top_10_rf['Feature'].map(mapping_dict).fillna(top_10_rf['Feature'])

print("--- Random Forest Top 10 Feature Importance ---")
print(top_10_rf[['Factor', 'Importance']])
```

    --- Random Forest Top 10 Feature Importance ---
                                    Factor  Importance
    14                 Credit_Success_Rate    0.326042
    15                           Avg_Grade    0.185061
    7   Financial: Tuition Paid up to Date    0.072186
    12                         Grade_Trend    0.052337
    10      Demographic: Age at Enrollment    0.044631
    6        Financial: Debt Owed (Debtor)    0.025143
    13                       Approved_Diff    0.024789
    9        Financial: Scholarship Holder    0.024504
    3                      Admission grade    0.023468
    2       Previous qualification (grade)    0.021039


**For Random Forest, the top 3 are our engineered features!!!**

# **Part 5: Live Demo Predictor**


```python
def live_predictor_hku_style():
    print("--- 🎓 Student Success Prediction (HKU Edition) ---")
    print("Please enter the academic details (GPA 4.3 Scale):")

    try:
        # 1. Inputs using local campus standards
        total_courses = float(input("How many courses did you take this year? (e.g., 10): "))
        passed_courses = float(input("How many courses did you pass? (e.g., 8): "))

        gpa_sem1 = float(input("GPA in Semester 1 (0 - 4.3): "))
        gpa_sem2 = float(input("GPA in Semester 2 (0 - 4.3): "))

        age = float(input("Age at enrollment (e.g., 18): "))
        tuition_val = float(input("Tuition fees settled? (1: Yes, 0: No): "))

        # 2. Conversion Logic to match original dataset (0-20 scale)
        # Mapping GPA 4.3 to 20 scale
        score_sem1 = (gpa_sem1 / 4.3) * 20
        score_sem2 = (gpa_sem2 / 4.3) * 20

        # Mapping Credits (6 units per course)
        # Note: Success Rate remains the same regardless of multiplier
        success_rate = passed_courses / total_courses if total_courses > 0 else 0
        avg_grade = (score_sem1 + score_sem2) / 2
        grade_trend = score_sem2 - score_sem1

        # 3. Prepare Feature Vector
        sample_data = X_train.mean().to_frame().T

        # 4. Fill with converted values
        sample_data['Credit_Success_Rate'] = success_rate
        sample_data['Avg_Grade'] = avg_grade
        sample_data['Grade_Trend'] = grade_trend
        sample_data['Age at enrollment'] = age
        sample_data['Tuition fees up to date'] = tuition_val

        # 5. Scaling and Prediction
        sample_scaled_array = scaler.transform(sample_data)
        sample_scaled_df = pd.DataFrame(sample_scaled_array, columns=X_train.columns)

        prediction = rf_model.predict(sample_scaled_df)[0]
        probability = rf_model.predict_proba(sample_scaled_df)[0]

        # 6. Output mapping
        target_map = {0: "DROPOUT (High Risk)",
                      1: "SUCCESS (Stay / Graduate)"}

        print("-"*50)
        print(f"Calculated to fit European model:")
        print(f"Average Grade (20 Credit System): {avg_grade:.2f}/20")
        print(f"Course Pass Rate: {success_rate*100:.1f}% ({int(passed_courses)}/{int(total_courses)})")
        print("\n" + "*" * 50)
        print("               ASSESSMENT RESULT")
        print("*" * 50)
        print(f"Final Prediction: {target_map[prediction]}")
        print(f"Model Confidence: {probability[prediction]*100:.2f}%")
        print("*"*50)

    except ValueError:
        print("Error: Please check your input format.")

# Run the localized predictor
live_predictor_hku_style()
```

    --- 🎓 Student Success Prediction (HKU Edition) ---
    Please enter the academic details (GPA 4.3 Scale):
    How many courses did you take this year? (e.g., 10): 10
    How many courses did you pass? (e.g., 8): 10
    GPA in Semester 1 (0 - 4.3): 2.0
    GPA in Semester 2 (0 - 4.3): 2.2
    Age at enrollment (e.g., 18): 19
    Tuition fees settled? (1: Yes, 0: No): 1
    --------------------------------------------------
    Calculated to fit European model:
    Average Grade (20 Credit System): 9.77/20
    Course Pass Rate: 100.0% (10/10)
    
    **************************************************
                   ASSESSMENT RESULT
    **************************************************
    Final Prediction: SUCCESS (Stay / Graduate)
    Model Confidence: 61.40%
    **************************************************


**Extra: XGBoost for binary classification**<br>
Target Class Consolidation: Dropout (0) and Success (1)


```python
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Initialize and train the binary model
weights = compute_sample_weight(class_weight='balanced', y=y_train_binary)
xgb_binary = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='binary:logistic', # Changed to binary
    random_state=42,
    eval_metric='logloss'
)

xgb_binary.fit(X_train, y_train_binary)

# 3. Evaluation for Binary Classification
y_pred_binary = xgb_binary.predict(X_test)
target_names_binary = ['Dropout', 'Success']

print(f"--- Binary XGBoost Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=target_names_binary))

# 4. Confusion Matrix Plot
plt.figure(figsize=(7, 5))
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Greens',
            xticklabels=target_names_binary,
            yticklabels=target_names_binary)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: XGBoost')
plt.show()
```

    --- Binary XGBoost Evaluation ---
    Accuracy: 0.8678
    
    Classification Report:
                  precision    recall  f1-score   support
    
         Dropout       0.83      0.74      0.78       284
         Success       0.88      0.93      0.91       601
    
        accuracy                           0.87       885
       macro avg       0.86      0.83      0.84       885
    weighted avg       0.87      0.87      0.87       885
    



    
![png](3544_Project_files/3544_Project_52_1.png)
    



```python
# Extracting importance for the binary model
xgb_bin_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_binary.feature_importances_
})

# Mapping descriptions
xgb_bin_importance['Description'] = xgb_bin_importance['Feature'].map(mapping_dict).fillna(xgb_bin_importance['Feature'])

# Sort and display Top 10
top_10_binary = xgb_bin_importance.sort_values(by='Importance', ascending=False).head(10)

print("=== Top 10 Features: Binary Classification ===")
print(top_10_binary[['Description', 'Importance']])
```

    === Top 10 Features: Binary Classification ===
                                           Description  Importance
    14                             Credit_Success_Rate    0.186880
    38       Course 171: Animation & Multimedia Design    0.079622
    7               Financial: Tuition Paid up to Date    0.074054
    120           Dad's Edu: Higher Education (Degree)    0.021970
    13                                   Approved_Diff    0.020101
    52                     Course 9853: Social Service    0.019565
    10                  Demographic: Age at Enrollment    0.017365
    6                    Financial: Debt Owed (Debtor)    0.016067
    87            Mom's Edu: Higher Education (Degree)    0.015567
    44   Course 9130: Equiniculture (Horse Management)    0.015066



```python
import pandas as pd
import numpy as np

def run_academic_risk_demo(model, feature_names, X_train):
    print("-" * 60)
    print("      GRADUATE VS. DROPOUT RISK ASSESSMENT")
    print("-" * 60)

    # Start with a baseline (median student)
    user_data = X_train.median().to_dict()

    print("Please provide student details:")

    # 1. Automatic Credit Rate Calculation
    try:
        taken = float(input("Number of courses enrolled last term (e.g., 5): ") or 5)
        passed = float(input("Number of courses passed last term (e.g., 4): ") or 4)
        user_data['Credit_Success_Rate'] = passed / taken if taken > 0 else 0
    except ValueError:
        user_data['Credit_Success_Rate'] = 0.8 # Fallback

    # 2. Automatic Mature Student Logic
    try:
        age = float(input("Student's age at enrollment (e.g., 18): ") or 18)
        user_data['Age at enrollment'] = age
        # Automatically determine "Mature Student" status
        user_data['Application mode_39'] = 1 if age > 23 else 0
    except ValueError:
        user_data['Age at enrollment'] = 18
        user_data['Application mode_39'] = 0

    # 3. Direct Binary Questions (Simple Yes/No)
    questions = [
        ('Tuition fees up to date', 'Are tuition fees fully paid? (1: Yes, 0: No): '),
        ('Scholarship holder', 'Is the student a scholarship holder? (1: Yes, 0: No): '),
        ('Debtor', 'Does the student have any outstanding debt? (1: Yes, 0: No): ')
    ]

    for feat, q in questions:
        val = input(q)
        user_data[feat] = float(val) if val.strip() != "" else user_data[feat]

    # 4. Major Selection
    print("\nSelect Major: [0] General [1] Animation [2] Social Service [3] Nursing")
    m_choice = input("Choice: ")
    for m in ['Course_171', 'Course_9853', 'Course_9500']: user_data[m] = 0
    if m_choice == '1': user_data['Course_171'] = 1
    elif m_choice == '2': user_data['Course_9853'] = 1
    elif m_choice == '3': user_data['Course_9500'] = 1

    # Final Prediction
    input_df = pd.DataFrame([user_data])[feature_names]
    prob = model.predict_proba(input_df)[0]
    pred = model.predict(input_df)[0]

    print("\n" + "="*40)
    status = "Success (Stay)" if pred == 1 else "Dropout (High Risk)"
    confidence = prob[1] if pred == 1 else prob[0]
    print(f"Prediction: {status}")
    print(f"Confidence: {confidence*100:.2f}%")
    print("="*40)

# run_academic_risk_demo(xgb_binary, X_train.columns, X_train)
```

# **ROC**


```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_pr_curve(models, X_test, y_test):
    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        # 獲取 Dropout (Class 0) 的預測機率
        # 注意：如果你的 y_test 裡 0 是 Dropout，機率通常是 probs[:, 0]
        # 但有些 Scikit-learn 函數預設看 Class 1，需確認你的編碼
        probs = model.predict_proba(X_test)[:, 0]

        # 計算 PR 數據 (這裡假設我們關注的是 Class 0: Dropout)
        # 如果要算 Dropout 的 Recall，y_true 需要轉換為: Dropout=1, Success=0
        y_true_dropout = (y_test == 0).astype(int)

        precision, recall, _ = precision_recall_curve(y_true_dropout, probs)
        ap_score = average_precision_score(y_true_dropout, probs)

        plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.2f})')

    plt.xlabel('Recall (Catching Dropouts)')
    plt.ylabel('Precision (Accuracy of Dropout Flags)')
    plt.title('Precision-Recall Curve for Dropout Prediction')
    plt.legend(loc='best')
    plt.show()

# 使用方法：
models_dict = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_binary
}
plot_pr_curve(models_dict, X_test, y_test_binary)
```


    
![png](3544_Project_files/3544_Project_56_0.png)
    



```python
!jupyter nbconvert --to markdown "3544 Project".ipynb
```

    [NbConvertApp] WARNING | pattern '3544 Project.ipynb' matched no files
    This application is used to convert notebook files (*.ipynb)
            to various other formats.
    
            WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.
    
    Options
    =======
    The options below are convenience aliases to configurable class-options,
    as listed in the "Equivalent to" description-line of the aliases.
    To see all configurable class-options for some <cmd>, use:
        <cmd> --help-all
    
    --debug
        set log level to logging.DEBUG (maximize logging output)
        Equivalent to: [--Application.log_level=10]
    --show-config
        Show the application's configuration (human-readable format)
        Equivalent to: [--Application.show_config=True]
    --show-config-json
        Show the application's configuration (json format)
        Equivalent to: [--Application.show_config_json=True]
    --generate-config
        generate default config file
        Equivalent to: [--JupyterApp.generate_config=True]
    -y
        Answer yes to any questions instead of prompting.
        Equivalent to: [--JupyterApp.answer_yes=True]
    --execute
        Execute the notebook prior to export.
        Equivalent to: [--ExecutePreprocessor.enabled=True]
    --allow-errors
        Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.
        Equivalent to: [--ExecutePreprocessor.allow_errors=True]
    --stdin
        read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'
        Equivalent to: [--NbConvertApp.from_stdin=True]
    --stdout
        Write notebook output to stdout instead of files.
        Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]
    --inplace
        Run nbconvert in place, overwriting the existing notebook (only
                relevant when converting to notebook format)
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]
    --clear-output
        Clear output of current file and save in place,
                overwriting the existing notebook.
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]
    --coalesce-streams
        Coalesce consecutive stdout and stderr outputs into one stream (within each cell).
        Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]
    --no-prompt
        Exclude input and output prompts from converted document.
        Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]
    --no-input
        Exclude input cells and output prompts from converted document.
                This mode is ideal for generating code-free reports.
        Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]
    --allow-chromium-download
        Whether to allow downloading chromium if no suitable version is found on the system.
        Equivalent to: [--WebPDFExporter.allow_chromium_download=True]
    --disable-chromium-sandbox
        Disable chromium security sandbox when converting to PDF..
        Equivalent to: [--WebPDFExporter.disable_sandbox=True]
    --show-input
        Shows code input. This flag is only useful for dejavu users.
        Equivalent to: [--TemplateExporter.exclude_input=False]
    --embed-images
        Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.
        Equivalent to: [--HTMLExporter.embed_images=True]
    --sanitize-html
        Whether the HTML in Markdown cells and cell outputs should be sanitized..
        Equivalent to: [--HTMLExporter.sanitize_html=True]
    --log-level=<Enum>
        Set the log level by value or name.
        Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']
        Default: 30
        Equivalent to: [--Application.log_level]
    --config=<Unicode>
        Full path of a config file.
        Default: ''
        Equivalent to: [--JupyterApp.config_file]
    --to=<Unicode>
        The export format to be used, either one of the built-in formats
                ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']
                or a dotted object name that represents the import path for an
                ``Exporter`` class
        Default: ''
        Equivalent to: [--NbConvertApp.export_format]
    --template=<Unicode>
        Name of the template to use
        Default: ''
        Equivalent to: [--TemplateExporter.template_name]
    --template-file=<Unicode>
        Name of the template file to use
        Default: None
        Equivalent to: [--TemplateExporter.template_file]
    --theme=<Unicode>
        Template specific theme(e.g. the name of a JupyterLab CSS theme distributed
        as prebuilt extension for the lab template)
        Default: 'light'
        Equivalent to: [--HTMLExporter.theme]
    --sanitize_html=<Bool>
        Whether the HTML in Markdown cells and cell outputs should be sanitized.This
        should be set to True by nbviewer or similar tools.
        Default: False
        Equivalent to: [--HTMLExporter.sanitize_html]
    --writer=<DottedObjectName>
        Writer class used to write the
                                            results of the conversion
        Default: 'FilesWriter'
        Equivalent to: [--NbConvertApp.writer_class]
    --post=<DottedOrNone>
        PostProcessor class used to write the
                                            results of the conversion
        Default: ''
        Equivalent to: [--NbConvertApp.postprocessor_class]
    --output=<Unicode>
        Overwrite base name use for output files.
                    Supports pattern replacements '{notebook_name}'.
        Default: '{notebook_name}'
        Equivalent to: [--NbConvertApp.output_base]
    --output-dir=<Unicode>
        Directory to write output(s) to. Defaults
                                      to output to the directory of each notebook. To recover
                                      previous default behaviour (outputting to the current
                                      working directory) use . as the flag value.
        Default: ''
        Equivalent to: [--FilesWriter.build_directory]
    --reveal-prefix=<Unicode>
        The URL prefix for reveal.js (version 3.x).
                This defaults to the reveal CDN, but can be any url pointing to a copy
                of reveal.js.
                For speaker notes to work, this must be a relative path to a local
                copy of reveal.js: e.g., "reveal.js".
                If a relative path is given, it must be a subdirectory of the
                current directory (from which the server is run).
                See the usage documentation
                (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)
                for more details.
        Default: ''
        Equivalent to: [--SlidesExporter.reveal_url_prefix]
    --nbformat=<Enum>
        The nbformat version to write.
                Use this to downgrade notebooks.
        Choices: any of [1, 2, 3, 4]
        Default: 4
        Equivalent to: [--NotebookExporter.nbformat_version]
    
    Examples
    --------
    
        The simplest way to use nbconvert is
    
                > jupyter nbconvert mynotebook.ipynb --to html
    
                Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].
    
                > jupyter nbconvert --to latex mynotebook.ipynb
    
                Both HTML and LaTeX support multiple output templates. LaTeX includes
                'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and
                'classic'. You can specify the flavor of the format used.
    
                > jupyter nbconvert --to html --template lab mynotebook.ipynb
    
                You can also pipe the output to stdout, rather than a file
    
                > jupyter nbconvert mynotebook.ipynb --stdout
    
                PDF is generated via latex
    
                > jupyter nbconvert mynotebook.ipynb --to pdf
    
                You can get (and serve) a Reveal.js-powered slideshow
    
                > jupyter nbconvert myslides.ipynb --to slides --post serve
    
                Multiple notebooks can be given at the command line in a couple of
                different ways:
    
                > jupyter nbconvert notebook*.ipynb
                > jupyter nbconvert notebook1.ipynb notebook2.ipynb
    
                or you can specify the notebooks list in a config file, containing::
    
                    c.NbConvertApp.notebooks = ["my_notebook.ipynb"]
    
                > jupyter nbconvert --config mycfg.py
    
    To see all available configurables, use `--help-all`.
    

