# Week 6 Progress Report

*   DISCUSSED WEEKLY WORKFLOW ALIGNING PROJECT TIMELINE
*   DATA EXPLORATION
    *   Bivariate analysis Cardio with scaled data set by dividing them into numerical features, categorical features, and binary features. Created Boxplot graph for numerical, scatter and stacked graphs for categorical, and stacked graph for binary.
    *   Univariate analysis with different columns and created Histogram for individual plotting
    *   Scaling without Binary columns: Scaled the dataset by removing the binary columns present in the dataset from the Data Frame using Python and created another data set named "scaled\_without\_bin.csv" with the scaled values merged with the binary columns removed before scaling.
    *   Merged the trained data sets. Scaled the dataset using the min-max scaling and created a data set named "scaled\_dataset.csv"
    *   Separated 'id' and 'cardio' columns into a new DataFrame (id\_cardio). Numerical features are then standardized using standard scaling excluding the 'id' and 'cardio' columns. Created a new data set using the scaled data set and the excluded columns named "scaled\_standard\_dataset.csv" and a success message is printed along with the display of the first row of the resulting DataFrame (result\_df).
    *   Performed co-relation on standard scaled data, visualized multi-collinearity, extracted features, viewed multi-collinearity b/w them, created a new dataset with raw features, visualized, created module file to use for visualization
    *   Performed multi-corelation on min\_max scaling dataset to find features based on different co-relation methods
*   DISCUSSED OHE AND SCALING TECHNIQUES
*   DISCUSSED DIFFERENT METHODS OF CO-RELATION

  

CONTRIBUTIONS

| **NAMES** | **TASKS** |
| ---| --- |
| Bikky Singh | SCALING - STANDARD SCALING |
| Shittu | MULTI COLLINEARITY - |
| Rutul | EDA - Univariate Analysis |
| Jankiba | EDA - Univariate Analysis |
| Gaurav Singh | EDA - Bivariate Analysis - Scaled Data | VERSION CONTROL |
| Prakash | DATA PREP - SCALING - minMAX SCALING |
| Tirth | MULTI COLLINEARITY - Scaled Data |
| Swetha | SCALING - Scaling without Binary columns |
| Aman Nain | MULTI COLLINEARITY - Standard Scaled Data<br>FEATURE EXTRACTION |
| EVERYONE | Try to build relationships with the Target Variable |