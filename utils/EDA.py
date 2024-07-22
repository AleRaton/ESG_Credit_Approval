import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    


def EDA(df, feature, target):
    bins = 30
    
    # Big title for the feature
    plt.figure(figsize=(12, 2))
    plt.text(0.5, 0.5, f'Exploratory Data Analysis for {feature}', 
             horizontalalignment='center', 
             verticalalignment='center', 
             fontsize=24, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    # Distribution of the feature
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], bins=bins)
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()
    
    # Boxplot of the feature against the target
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}', fontsize=16)
    plt.xlabel(feature)
    plt.show()
    
    # Mean of the feature over time
    df['Year'] = pd.to_datetime(df['Year'])
    mean = df.set_index('Year').resample('YE')[feature].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y=feature, data=mean)
    plt.title(f'Evolution of the Mean of {feature} Over Time (by Year)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel(f'Mean of {feature}')
    plt.show()
    
    # Missing values of the feature over time
    missing_values_count = df.set_index('Year').resample('YE')[feature].apply(lambda x: x.isnull().sum()).reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Year', y=feature, data=missing_values_count)
    plt.title(f'Evolution of Missing Values of {feature} Over Time (by Year)', fontsize=16)
    plt.xlabel('Year')
    plt.ylabel(f'Count of Missing {feature}')
    plt.show()

# Example usage:
# EDA(your_dataframe, 'your_feature', 'your_target')

# Example usage
# EDA(df, 'feature_name', 'target_name')

def correlation_matrix(df):  
    plt.figure(figsize=(20,15))
    sns.heatmap(df.corr().round(2), annot=True,cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show() 