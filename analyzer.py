import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataAnalyzer:
    def __init__(self, X, y):
        data = pd.concat([X, y], axis=1)
        data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})
        self.data = data
        self.target = y.columns[0]

    def correlation_matrix(self):
        corr = self.data.corr(method='pearson')
        fig = px.imshow(corr, color_continuous_scale=px.colors.diverging.RdBu)
        fig.show()

    def perform_univariate_analysis(self):
        columns_to_analyze = self.data.columns.difference([self.target])
        results = []

        for column in columns_to_analyze:
            malignant = self.data[self.data[self.target] == 1][column]
            benign = self.data[self.data[self.target] == 0][column]
            t_statistic, p_value = stats.ttest_ind(malignant, benign)
            results.append({
                'Feature': column,
                'T-Statistic': t_statistic,
                'P-Value': p_value
            })

        t_test_results = pd.DataFrame(results)
        print(t_test_results)

    def analyze_columns(self):
        columns_to_analyze = self.data.columns.difference([self.target])
        for column in columns_to_analyze:
            summary_stats = self.data[column].describe()
            print(f"Summary Statistics for {column}:\n{summary_stats}\n")

            plt.figure(figsize=(8, 6))
            sns.histplot(data=self.data, x=column, kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.show()

            plt.figure(figsize=(8, 6))
            sns.boxplot(data=self.data, x=column)
            plt.title(f"Box Plot of {column}")
            plt.xlabel(column)
            plt.show()
