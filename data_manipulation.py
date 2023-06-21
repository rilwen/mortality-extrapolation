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
    min_age = cohort_rates.columns.min()
    min_year_birth = cohort_rates.index.min()
    period_years = cohort_rates.index + min_age
    period_rates = pd.DataFrame(np.nan, index=period_years, columns=cohort_rates.columns)
    for age in cohort_rates.columns:
        values = cohort_rates[age].dropna().values
        period_rates.loc[(min_year_birth + age):, age] = values
    return period_rates
    

def period_to_cohort_rates(period_rates):
    """Period rates index = period year
    Cohort rates columns = age at giving birth"""
    min_age = period_rates.columns.min()
    years_of_birth = period_rates.index - min_age
    cohort_rates = pd.DataFrame(np.nan, index=years_of_birth, columns=period_rates.columns)
    for age in period_rates.columns:
        known_period_rates = period_rates[age].dropna()
        min_year = known_period_rates.index.min()
        max_year = known_period_rates.index.max()
        cohort_rates.loc[(min_year - age):(max_year - age), age] = known_period_rates.values
    return cohort_rates