import numpy as np
import pandas as pd

def load_data(path):
    """Load data from path."""
    df = pd.read_csv(path, parse_dates=False, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df
    

def cohort_to_period_rates(cohort_rates):
    """Cohort rates index = year of birth
    Cohort rates columns = age at giving birth"""
    min_age = cohort_rates.index.min()
    min_year_birth = cohort_rates.columns.min()
    period_years = cohort_rates.columns + min_age
    period_rates = pd.DataFrame(np.nan, columns=period_years, index=cohort_rates.index)
    for age in cohort_rates.index:
        values = cohort_rates.loc[age].dropna().values
        period_rates.loc[age, (min_year_birth + age):] = values
    return period_rates
    

def period_to_cohort_rates(period_rates):
    """Period rates index = period year
    Cohort rates columns = age at giving birth"""
    min_age = period_rates.index.min()
    years_of_birth = period_rates.columns - min_age
    cohort_rates = pd.DataFrame(np.nan, columns=years_of_birth, index=period_rates.index)
    for age in period_rates.index:
        known_period_rates = period_rates[age].dropna()
        min_year = known_period_rates.columns.min()
        max_year = known_period_rates.columns.max()
        cohort_rates.loc[age, (min_year - age):(max_year - age)] = known_period_rates.values
    return cohort_rates