import pandas as pd
from scipy.stats import kstest,f_oneway,chi2_contingency,kruskal,pointbiserialr,stats

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro 
from scipy.stats import kstest
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import probplot
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison

def cat_bmi(bmi):
    """
    Revoie la categorie dans laquelle le bmi ce situe
    """
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "healthy"
    elif bmi <30:
        return "overweight"
    elif bmi < 40:
        return "obesity"
    else:
        return "morbid_obesity"

def point_biserial_correlation(data: pd.DataFrame, x_col: str, y_col: str, alpha: float = 0.05) -> None:
    """
    Calculate the point biserial correlation coefficient between two variables in a Pandas DataFrame.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the two variables.
    x_col (str): The name of the continuous variable.
    y_col (str): The name of the dichotomous variable.
    alpha (float): The significance level (default is 0.05).
    
    Returns:
    None

    Example:
    >>> data = pd.DataFrame({'charges': [1000, 2000, 3000, 4000, 5000], 'sex': ['male', 'male', 'female', 'female', 'male']})
    >>> point_biserial_correlation(data, 'charges', 'sex', alpha=0.05)
    ----------------------------
    | Correlation between "charges" and "sex" |
    ----------------------------
    There is a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.
    
    """
    # Extract the variables from the DataFrame
    x = data[x_col]
    y = data[y_col]
    
    # Convert the dichotomous variable to a list of 0s and 1s
    y = [0 if yi == y.unique()[0] else 1 for yi in y]
    x = [xi for xi in x]  
   
    # Calculate the point biserial correlation coefficient and p-value
    rpb, p_value = stats.pointbiserialr(x, y)
    
   # Split the continuous variable data into two groups based on the dichotomous variable
    x_group1 = [x[i] for i in range(len(y)) if y[i] == 0]
    x_group2 = [x[i] for i in range(len(y)) if y[i] == 1]

    
    # Perform the t-test
    t, p_value_ttest = stats.ttest_ind(x_group1, x_group2)
    
    # Print the title in a boxed format
    title = f'Correlation between "{x_col}" and "{y_col}"'
    print('-' * (len(title) + 4))
    print(f'| {title} |')
    print('-' * (len(title) + 4))
    print(f'Point biserial correlation coefficient: {rpb:.3f}')
    print(f't-value: {t:.3f}')
    print(f'p-value: {p_value}')
    
    # Print the interpretation of the p-value
    if p_value_ttest < alpha:
        print('There is a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.')
    else:
        print('There is not a significant difference in the means of the continuous variable between the two groups defined by the dichotomous variable.')


def get_index_to_remove_by_Cooks_Distance(X_train, y_train, preprocessor,seuil_dcook =0.005):
    """
    This function removes observations from the training data that have high Cook's distance values.
    Cook's distance is a measure of the influence of an observation on a statistical model.
    Observations with high Cook's distance values may have a disproportionate influence on the model,
    and removing them can improve the model's accuracy.
    
    Parameters:
    - X_train: pd.DataFrame
        The training data
    - y_train: pd.Series
        The target labels for the training data
    - preprocessor: sklearn.compose.ColumnTransformer
        The preprocessor created by make_pipeline_to_ML()
        
    Returns:
    - index_to_be_removed: pd.Index
        The indices of the observations to be removed from the training data
    """
    
    # Fit the transformer to the training data
    preprocessor.fit(X_train)
    
    # Transform the training data using the preprocessor
    X_test_pipe = preprocessor.transform(X_train)
    
    # Get the names of the columns added by the OneHotEncoder
    new_columns = preprocessor.get_feature_names_out()
    new_columns = [w.replace('pipeline-1__', '') for w in new_columns]
    new_columns = [w.replace('pipeline-2__', '') for w in new_columns]
    
    # Convert the transformed data to a Pandas DataFrame
    newdf = pd.DataFrame(X_test_pipe)
    
    # Set the column names to the names obtained from the OneHotEncoder
    newdf.columns = new_columns
    
    # Add a constant term to the DataFrame
    X = sm.add_constant(newdf)
    
    # Set the index of the DataFrame to the index of the target labels
    X = X.set_index(y_train.index)
    
    # Fit an OLS model to the data
    estimation = sm.OLS(y_train, X_test_pipe).fit()
    
    # Calculate the Cook's distance values for each observation
    influence = estimation.get_influence().cooks_distance[0]
    
    # Add the Cook's distance values to the DataFrame as a new column
    X['dcooks'] = influence
    
    # Calculate the threshold for Cook's distance values
    n = X.shape[0]
    p = X.shape[1]
    
    
    # Select the indices of the observations with Cook's distance values above the threshold
    index_to_be_removed = X[X['dcooks']>seuil_dcook].index
    
    # Return the indices of the observations to be removed
    plt.figure(figsize=(20,8))
    plt.bar(X.index, X['dcooks'])
    plt.xticks(np.arange(0, len(X), step=int(len(X)/10)))
    plt.xlabel('Observation')
    plt.ylabel('Cooks Distance')
    #Plot the line
    plt.hlines(seuil_dcook, xmin=0, xmax=len(X_train), color='r')
    plt.show()
    return index_to_be_removed

