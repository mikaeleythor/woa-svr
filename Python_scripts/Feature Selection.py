#Feature Selection
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# hlaða inn
# breyta árum eftir þörfum

file_path = r'C:\Users\Gunnar Ólafsson\OneDrive - Reykjavik University\Desktop\Location4_normalized_z_norm.csv'
df = pd.read_csv(file_path)
# taka tima dalkinn ut þurfum hann ekki.
df_numeric = df.drop(columns=['Time'])
# correlation and analysis reyna að skrifa ut correlation matrixum i text formatii
correlation_matrix = df_numeric.corr()
print("Correlation matrix:")
print(correlation_matrix)
# einblinum á correlation vid breyturnar og targetið power
print("\nCorrelation with Power:")
# skrifa ut eftir hæstu til lægstu
print(correlation_matrix['Power'].sort_values(ascending=False))
# plotta upp heatmap.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()
# mogulega cutta a þetta heðan 
# decision treeið gefur mjög misvisandi svör miðað við correlation with power ut fra þvi að hvað correlation matrixið er
# kikjum a það seinna gæti verið að RFR gæti verið að sja fram a eitthvað non linear samband en breyturnar eru bara svo lágar.
# Generate Pair Plot (scatter plot matrix)
#sns.pairplot(df_numeric)
#plt.suptitle('Scatter Plot Matrix', y=1.02)
#plt.show()

# plotta hisgrogram á featureanana 
df_numeric.hist(bins=20, figsize=(14, 10))
plt.suptitle('Histograms of Numeric Features', y=1.02)
plt.tight_layout()
plt.show()

# skilgriena featurs og target breytuna
X = df_numeric.drop('Power', axis=1)
y = df_numeric['Power']

# profa að traina random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# get feature importance fyrir histogram.
importances = rf.feature_importances_

# plotta uppp i histogram.
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance Using Random Forest')
plt.show()
