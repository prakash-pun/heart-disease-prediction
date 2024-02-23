import pandas as pd
import matplotlib.pyplot as plt
from utils import get_data  

df = get_data()

# Display basic information about the dataset
print(df.head())
print(df.info())

# Extract the 'smoke', 'alco', and 'cardio' columns
selected_columns = df[['smoke', 'alco', 'cardio']]

# Display the correlation matrix
correlation_matrix = selected_columns.corr()
print("Correlation Matrix:\n", correlation_matrix)

# Plot a simple heatmap to visualize the correlation
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Heatmap: Smoke, Alcohol, and Cardio')
plt.show()
