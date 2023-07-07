import pandas as pd
import os
import requests
import re
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, MaxAbsScaler

# Checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# Download data if it is unavailable.
if 'nba2k-full.csv' not in os.listdir('../Data'):
    print('Train dataset loading.')
    url = "https://www.dropbox.com/s/wmgqf23ugn9sr3b/nba2k-full.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/nba2k-full.csv', 'wb').write(r.content)
    print('Loaded.')

data_path = "../Data/nba2k-full.csv"


def clean_data(data_path):

    def process_height(height):
        return re.sub(".*/ ", "", height)

    def process_weight(weight):
        return weight.split("/")[1].replace(r" kg.", "")

    def process_salary(salary):
        return salary.replace(r"$", "")

    df = pd.read_csv(data_path)
    df["b_day"] = pd.to_datetime(df["b_day"], format="%m/%d/%y")
    df["draft_year"] = pd.to_datetime(df["draft_year"], format="%Y")
    df["team"].fillna("No Team", inplace=True)
    df["height"] = df["height"].apply(process_height)
    df["weight"] = df["weight"].apply(process_weight)
    df["salary"] = df["salary"].apply(process_salary)
    df["height"] = df["height"].astype(float)
    df["weight"] = df["weight"].astype(float)
    df["salary"] = df["salary"].astype(float)
    df.loc[df["country"] != "USA", "country"] = "Not-USA"
    df.loc[df["draft_round"] == "Undrafted", "draft_round"] = "0"
    return(df)


def feature_data(df):
    def process_version(txt):
        return pd.to_datetime(txt[-2:], format="%y")

    df["version"] = pd.to_datetime(
        df.version.apply(process_version).
        dt.strftime(date_format="%Y"), format="%Y")
    df["age"] = df["version"].dt.year - df["b_day"].dt.year
    df["age"] = df["age"].astype(int)
    df["experience"] = df["version"].dt.year - df["draft_year"].dt.year
    df["experience"] = df["experience"].astype(int)
    df["bmi"] = df["weight"] / df["height"]**2
    # removing previous variables already engineered
    columns_to_drop = ["version", "b_day", "draft_year", "weight", "height"]
    keep_columns = df.columns[~df.columns.isin(columns_to_drop)]
    df = df[keep_columns]
    # removing high cardinality categorical features
    columns_to_drop = df.columns[(df.nunique() > 50) & (df.dtypes == "object")]
    keep_columns = df.columns[~df.columns.isin(columns_to_drop)]
    df = df[keep_columns]
    return df


def VIF_table(X_variables):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_variables.columns
    vif_data["VIF"] = [variance_inflation_factor(X_variables.values, i) for i in range(len(X_variables.columns))]
    vif_data.sort_values(by="VIF", ascending=False, inplace=True, ignore_index=True)
    return vif_data


def combs(colnames):
    comb = list()
    for i in range(1, len(colnames)-1):
        comb.extend([j for j in combinations(colnames, i)])
    return comb


def heatmap(df, cmap="Reds", title=""):
    X_variables = df[df.columns[~(df.dtypes == object)]]
    labels = X_variables.corr().values
    for a in range(labels.shape[0]):
        for b in range(labels.shape[1]):
            plt.text(a, b, '{:.2f}'.format(labels[b, a]), ha='center', va='center', color='black')
    plt.xticks(range(len(X_variables.corr().columns)), X_variables.corr().columns)
    plt.yticks(range(len(X_variables.corr().columns)), X_variables.corr().columns)
    plt.title(title +' \n', fontsize=14)
    plt.imshow(X_variables.corr(), cmap=cmap)
    plt.colorbar()
    plt.show()


def multicol_data(df, exclude_low_cor=True):
    # exclude_low_cor: excludes the variables that generate a high VIF value but
    # do not have correlation values over 0.7.
    # exclude_targets prevents the column names from the list to be subject to deletion.
    X_variables = df[df.columns[~(df.dtypes == object)]]
    X_variables = X_variables[X_variables.columns[~(X_variables.columns.isin(["salary"]))]]
    ordered_vif = VIF_table(X_variables)
    collinear_vars = ordered_vif.loc[ordered_vif["VIF"] > 10, "feature"]
    collinear_combs = combs(collinear_vars)
    scores = dict()
    for i, comb in enumerate(collinear_combs):
        vif_elems = X_variables.columns[~X_variables.columns.isin(list(comb))]
        tbl = VIF_table(X_variables[vif_elems])
        scores[i] = [tbl.VIF.sum(), all(tbl.VIF < 10), comb]
    scores = pd.DataFrame(scores).T
    scores.columns = ["sum_score", "VIF_under_10", "removed_vars"]
    scores.sort_values(by=["VIF_under_10", "sum_score"], ascending=[False, True], inplace=True, ignore_index=False)
    best_score_index = scores.iloc[0, :].name
    vars_to_drop = collinear_combs[best_score_index]
    var_cor_tbl = X_variables.corr().unstack()
    high_cor_vars = list(set([i[0] for i in var_cor_tbl.loc[(var_cor_tbl >= 0.7) & (var_cor_tbl < 1)].index]))
    vars_to_drop = list(set(vars_to_drop).intersection(high_cor_vars))
    #print("vars dropped: ", vars_to_drop)
    reduced_df = df.drop(list(vars_to_drop), axis=1)
    return reduced_df

def transform_data(df):
    num_feat_df = df.select_dtypes('number')  # numerical features
    cat_feat_df = df.select_dtypes('object')  # categorical features
    target_feat = np.array(num_feat_df["salary"]).reshape(-1, 1)

    cols = num_feat_df.columns.tolist()
    cols.remove("salary")
    # StandardScaler standardizes features in columns independently.
    # StandardScaler returns an array in the same order as the input.Therefore, they
    # have the same names
    num_feat_scaled = pd.DataFrame(StandardScaler().fit_transform(num_feat_df[cols], target_feat),
                                   columns=cols)
    one_hot_ = OneHotEncoder(sparse_output=False) # return results as arrays
    cat_feats_encoded = pd.DataFrame(one_hot_.fit_transform(cat_feat_df, target_feat),
                             columns=[i for row in one_hot_.categories_ for i in row])
    result_df = pd.concat([num_feat_scaled, cat_feats_encoded], axis=1)
    return result_df, pd.Series(target_feat.tolist())


df_cleaned = clean_data(data_path)
df_featured = feature_data(df_cleaned)
df = multicol_data(df_featured)
# print(list(df.select_dtypes('number').drop(columns='salary')))
X, y = transform_data(df)

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }
print(answer)