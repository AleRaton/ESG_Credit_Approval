import seaborn as sns

def EDA(df,feature,target):
    bins = 30
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))

    # Histplot
    sns.histplot(df[feature], bins=bins, ax=axes[0,0])
    axes[0,0].set_title(f'Distribution of {feature}')

    # Boxplot 
    sns.boxplot(x=target,y=feature,data=df, ax=axes[1,0])
    axes[1,0].set_title(f'Boxplot of {feature}')

    # Mean over Time
    mean = df.groupby(pd.Grouper(key='Year', freq='YE'))[feature].mean().reset_index()
    sns.lineplot(x='Year', y=feature, data=mean, ax=axes[0,1])
    plt.xlabel('Year')
    plt.ylabel(f'Mean of {feature}')
    axes[0,1].set_title(f'Evolution of the mean of {feature} over time (by year)')
    
    # Missing Values over Time
    missing_values_count = df.groupby(pd.Grouper(key='Year', freq='YE'))[feature].apply(lambda x: x.isnull().sum()).reset_index()
    sns.lineplot(x='Year', y=feature, data=missing_values_count, ax=axes[1,1])
    plt.xlabel('Year')
    plt.ylabel(f'Count of missing {feature}')
    axes[1,1].set_title(f'Evolution of the missing values of {feature} over time (by year)')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def correlation_matrix(df):  
    plt.figure(figsize=(20,15))
    sns.heatmap(df.corr().round(2), annot=True,cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show() 