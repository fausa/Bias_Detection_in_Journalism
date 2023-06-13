# Import basic libraries
import pandas as pd
import numpy as np

# Import model and performance evaluation libraries
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

"""
This function is effectively a customized/extended describe(),
attempting to summarize key (numeric and categorical) characteristics
into a single view for preliminary understanding going into deeper
EDA and modeling.
"""
def profile(df):
    SKEW_TH = 3
    LNVAR_TH = 0.01
    VIF_TH = 10
    EXAMPLE_N = 5
    EXAMPLE_LEN = 20

    num_cols = df.select_dtypes('number').columns

    pf = pd.DataFrame(df.columns, columns=[None]).set_index(None)

    pf['Dtype'] = pd.DataFrame(df.dtypes)
    pf['count'] = df.count()
    pf['unique'] = df.nunique(axis=0)
    pf['na'] = df.isna().sum()
    pf['na%'] = pf['na'] / len(df) * 100
    pf['mean'] = pd.DataFrame(df[num_cols].mean())
    pf['std'] = pd.DataFrame(df[num_cols].std())
    pf['min'] = pd.DataFrame(df[num_cols].min())
    pf['max'] = pd.DataFrame(df[num_cols].max())

    pf['skew(>='+str(SKEW_TH)+')'] = pd.DataFrame(df[num_cols].skew()).apply(lambda x: abs(x*(x>=SKEW_TH)))

    lnvar_th_colname = '<v' + str(LNVAR_TH)
    nz_check = VarianceThreshold(LNVAR_TH).fit(df[num_cols])
    pf[lnvar_th_colname] = pd.concat([pd.DataFrame(df[num_cols].columns,
                                                   columns=[None]),
                                      pd.DataFrame(~nz_check.get_support(),
                                                   columns=[lnvar_th_colname])],
                                     axis=1).set_index(None)

    vif_th_colname = 'VIF(>=' + str(VIF_TH) + ')'
    with np.errstate(divide='ignore'):  # filter innocuous div by zero 'warning' VIF 
        pf[vif_th_colname] = pd.DataFrame(
            [[num_cols[i], variance_inflation_factor(np.nan_to_num(df[num_cols].values), i)]
            for i in range(len(df[num_cols].columns))],
            columns=[None, vif_th_colname]).set_index(None)
    pf[vif_th_colname] = pf[vif_th_colname].apply(lambda x: abs(x*(x>=VIF_TH)))

    pf = round(pf, 1).astype(str)
    pf.replace(['0', '0.0', 'nan', 'False'], '', inplace=True)

    edf = df.sample(n=EXAMPLE_N, axis=0).transpose().astype(str)
    pf['examples'] = edf.apply(lambda row: '__'.join(row.values.astype(str))[0:EXAMPLE_LEN], axis=1)

    display(pf)

"""
This function summarizes categories along with relative frequency (balance)
"""
def profile_cat(df, cat_cols):
    MAX_CV = 100

    cat_df = pd.DataFrame(columns=['cat_col', 'cat', 'freq'])
    cat_count = len(df)
    for cc in cat_cols:
        for cv in pd.unique(df[cc]):
            row = pd.DataFrame({'cat_col': [cc],
                                'cat': [cv],
                                'freq': [df[df[cc] == cv][cc].count() / cat_count * 100]})
            cat_df = pd.concat([cat_df, row])

    cat_df.sort_values(by=['cat_col', 'freq', 'cat'],
                       ascending=[True, False, True], inplace=True)
    for cc in cat_cols:
        print('\n', cc, ' - ', sep='')
        rows = cat_df[cat_df['cat_col'] == cc][['cat', 'freq']]
        print(rows.head(MAX_CV).to_string(header=False, index=False))
        if len(rows) > MAX_CV:
            print('..... and %0d more' % (len(rows) - MAX_CV))