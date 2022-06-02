import pandas as pd
import numpy as np
import csv

df_path = "Dataset/HCMST_ver_3.04.dta"
wave4_dfpath = 'Dataset/wave_4_supplement_v1_2.dta'
wave5_dfpath = 'Dataset/HCMST_wave_5_supplement_ver_1.dta'

def load_data(df_path):
    df = None
    with open(df_path, "r") as f:
        df = pd.read_stata(f)
    return df

def remove_rows(df):
    """ remove single examples """
    df = df[df['s2'].notnull() & df['s2'].str.contains('single') == False]
    df = df[df['s2'].notnull() & df['s2'].str.contains('refused') == False]
    return df

def create_label(df):
    """ Set label to 0 if couple broke up anytime until wave3, Otherwise 1 """
    df['y'] = np.where( (df['w2_q3'].notnull() & df['w2_q3'].str.contains('divorce') ) |
        (df['w2_q9'].notnull() & (df['w2_q9'].str.contains('deceased') == False)) |
        (df['w3_q3'].notnull() & df['w3_q3'].str.contains('divorce')) |
        (df['w4_q1'].notnull() & df['w4_q1'].str.contains('no')) |
        (df['w4_q5'].notnull() & df['w4_q5'].str.contains('no')) |
        (df['w234_combo_breakup'].notnull() & df['w234_combo_breakup'].str.contains('broke'))|
        (df['w5_broke_up'].notnull() & df['w5_broke_up'].str.contains('broke')) |
        (df['w2345_combo_breakup'].notnull() & df['w2345_combo_breakup'].str.contains('broke')), 0, 1)
    return df

def create_X(df):
    features = ['older','relationship_type','age_difference', 'q23', #who earns more
     'parental_approval', 'gender_attraction',
     'met_through_friends', 'met_through_family', 'met_through_as_neighbors',
     'met_through_as_coworkers', 'how_long_ago_first_met', 'how_long_relationship',
     'how_long_ago_first_romantic', 'how_long_ago_first_cohab', 'relationship_quality', 'children_in_hh',
     'number_of_marriages', 'edu_diff', 'edu_high', 'edu_low', 'edu_ave',
     'same_race', 'same_polit', 'same_religion', 'same_highschool',
     'same_college', 'same_city', 'parents_involved', 'internet', 'quality_change',
     'attractiveness_diff', 'sex_freq', 'w5_p_monogamy']

    df['edu_high'] = df[ ['respondent_yrsed', 'partner_yrsed'] ].max(axis=1)
    df['edu_low'] = df[ ['respondent_yrsed', 'partner_yrsed'] ].min(axis=1)
    df['edu_ave'] = ( df['respondent_yrsed'] + df['partner_yrsed'] ) / 2.0
    df['edu_diff'] = abs( df['respondent_yrsed'] - df['partner_yrsed'])
    df['same_race'] = np.where(df['respondent_race'] == df['partner_race'], 1, 0)

    df['papreligion'] = df.papreligion.str.replace(r'(^.*non.*$)', 'refused')
    df['same_religion'] = np.where(df['papreligion'] == df['q7b'], 1, 0)

    df['q12'] = df.q12.str.replace(r'(^.*no.*$)', 'other')
    df['q12'] = df.q12.str.replace(r'(^.*independent.*$)', 'other')
    df['q12'] = df.q12.str.replace(r'(^.*refused.*$)', 'other')
    df['same_polit'] = np.where(df['pppartyid3'] == df['q12'], 1, 0)

    # categorize into ranges
    df['age_difference'] = pd.cut(df.age_difference, bins=[-1,0,5,10,15,20,30,40,100], labels=[0,5,10,15,20,30,40,70])

    #homosexual
    cond1 = (df['q4'].str.contains('female') & df['ppgender'].str.contains('female')) | ((df['q4'].str.strip() == 'male') & (df['ppgender'].str.strip() == 'male'))
    #hetero
    cond2 = df['q4'].str.contains('female') & (df['ppgender'].str.strip() == 'male') & df['q23'].str.contains('i')
    df['q23'] = np.where(cond1, 0, np.where(cond2, 1, 2))


    approval_dict =  {"don't approve or don't know": 0, "approve" : 1 }
    df['parental_approval'] = df['parental_approval'].apply(lambda x: approval_dict[x])
    attract_dict = {'only same gender':0, 'same gender mostly':1, 'both genders equally':2, 'mostly opposite': 3, 'opposite gender only':4}
    df['gender_attraction'] = df['gender_attraction'].apply(lambda x: attract_dict[x])
    friends_dict =  {"not met through friends": 0, "meet through friends" : 1 }
    df['met_through_friends'] = df['met_through_friends'].apply(lambda x: friends_dict[x])
    family_dict =  {"not met through family": 0, "met through family" : 1 }
    df['met_through_family'] = df['met_through_family'].apply(lambda x: family_dict[x])
    neigh_dict =  {"did not meet through or as neighbors": 0, "met through or as neighbors" : 1 }
    df['met_through_as_neighbors'] = df['met_through_as_neighbors'].apply(lambda x: neigh_dict[x])

    df['how_long_ago_first_met'] = pd.cut(df.how_long_ago_first_met, bins=[-1,0,5,10,15,20,30,40,50], labels=[0,5,10,15,20,30,40,50])
    df['how_long_relationship'] = pd.cut(df.how_long_relationship, bins=[-1,0,5,10,15,20,30,40,50], labels=[0,5,10,15,20,30,40,50])
    df['how_long_ago_first_romantic'] = pd.cut(df.how_long_ago_first_romantic, bins=[-1,0,5,10,15,20,30,40,50,60,75], labels=[0,5,10,15,20,30,40,50, 60, 75])
    df['how_long_ago_first_cohab'] = pd.cut(df.how_long_ago_first_cohab, bins=[-1,0,5,10,15,20,30,40,50, 60, 70], labels=[0,5,10,15,20,30,40,50, 60, 70])

    quality_dict =  {"very poor": 0, "poor" : 1, "fair":2, "good":3, "excellent":4}
    qualityw4_dict ={ 'Refused': 5, "Very Poor": 0, "Poor" : 1, "Fair":2, "Good":3, "Excellent":4 }
    df['relationship_quality'] = df['relationship_quality'].apply(lambda x: quality_dict[x])
    # quality_change - higher means quality improvement
    df['relationship_quality'] = df['relationship_quality'].astype('float32')
    df['quality_new']= df['w4_quality'].apply(lambda x: qualityw4_dict[x])
    df['quality_new'] = df['quality_new'].astype('float32')
    df['quality_change'] = 4 + (df['quality_new'] - df['relationship_quality'])

    df['number_of_marriages'] = str_join(df, ' ', 'q17a', 'q17b')
    marriages_dict =  {"nan never married": 0, "nan once" : 1, "nan twice":2, "nan three times":3, "nan four or more times":4, "nan refused":5,
     "once (this is my first marriage) nan": 0, "twice nan": 1, "three times nan": 2, "four or more times nan": 3, "refused nan":5,
     "nan nan":5}
    df['number_of_marriages'] = df['number_of_marriages'].apply(lambda x: marriages_dict[x])

    df['relationship_type'] = str_join(df, ' ', 's1', 's2')
    relation_dict =  {"yes, i am married nan": 0, "no, i am not married yes, i have a sexual partner (boyfriend or girlfriend)": 1,
    "no, i am not married i have a romantic partner who is not yet a sexual partner": 2, "no, i am not married refused":3}
    df['relationship_type'] = df['relationship_type'].apply(lambda x: relation_dict[x])

    #if male older than female
    cond1 = ((np.asarray(df['ppage']) > np.asarray(df['q9'])) & (df['ppgender'].str.strip() == 'male')) | ((np.asarray(df['ppage']) < np.asarray(df['q9'])) & (df['ppgender'].str.contains('female')))
    cond2 = df['ppage'] == df['q9']

    #if male older than female = 1
    df['older'] = np.where(cond2, 0, np.where(cond1, 1, 2))

    #similarity in high school
    cond1 = df['q25'].str.contains('different')
    cond2 = df['q25'].str.contains('same')
    df['same_highschool'] = np.where(cond1, 0, np.where(cond2, 1, 2))
    #college
    cond1 = df['q26'].str.contains('did not attend')
    cond2 = df['q26'].str.contains('attended same college')
    df['same_college'] = np.where(cond1, 0, np.where(cond2, 1, 2))

    #city
    cond1 = df['q27'].str.contains('no')
    cond2 = df['q27'].str.contains('yes')
    df['same_city'] = np.where(cond1, 0, np.where(cond2, 1, 2))

    #parents know eachother
    cond1 = df['q28'].str.contains('no')
    cond2 = df['q28'].str.contains('yes')
    df['parents_involved'] = np.where(cond1, 0, np.where(cond2, 1, 2))

    #internet meet up
    cond1 = df['q32'].str.contains('no')
    cond2 = df['q32'].str.contains('matchmaking')
    cond3 = df['q32'].str.contains('yes')
    df['internet'] = np.where(cond1, 0, np.where(cond2, 1, np.where(cond3, 2, 3)))

    # attracttiveness
    attracttiveness_dict = { 'very attractive': 3, 'moderately attractive': 2, 'slightly attractive':1, 'not at all attractive': 0, 'Refused': 4 }
    df['attractive_1'] = df['w4_attractive'].apply(lambda x: attracttiveness_dict[x])
    df['attractive_2'] = df['w4_attractive_partner'].apply(lambda x: attracttiveness_dict[x])
    df['attractive_1'] = df['attractive_1'].astype('float32')
    df['attractive_2'] = df['attractive_2'].astype('float32')
    df['attractiveness_diff'] = abs(df['attractive_1']-df['attractive_2'])

    # sex frequency
    sex_freq_dict = {'Once a day or more': 4, '3 to 6 times a week': 3, 'Once or twice a week':2, '2 to 3 times a month': 1, 'Once a month or less':0, 'Refused':5}
    df['sex_freq'] = df['w5_sex_frequency'].apply(lambda x: sex_freq_dict[x])

    # monogamy
    cond1 = df['w5_p_monogamy'].str.contains('No')
    cond2 = df['w5_p_monogamy'].str.contains('Yes')
    df['w5_p_monogamy'] = np.where(cond1, 0, np.where(cond2, 1, 2))

    df = df[ [i for i in features] ]
    return df, features


def str_join(df, sep, *cols):
    from functools import reduce
    return reduce(lambda x, y: x.astype(str).str.cat(y.astype(str), sep=sep),[df[col] for col in cols])

def main():
    df = load_data(df_path)
    df_wave4 = load_data(wave4_dfpath)
    df_wave5 = load_data(wave5_dfpath)
    print 'The original size of each dataset is: '
    print df.shape, df_wave4.shape, df_wave5.shape
    df = pd.concat([df, df_wave4, df_wave5],axis=1)
    print 'The shape of dataset after concating:'
    print df.shape
    df = remove_rows(df)
    updateData(False, df)
    # obtain X
    X, Xnames = create_X(df)
    for f in Xnames:
        X[f] = X[f].astype('category')
        X[f] = X[f].cat.add_categories(-1).fillna(-1)
    npX = np.array(X)

    # print 'The size of X: '
    # print npX.shape

    # obtain y
    df = create_label(df)
    y = np.array( df['y'] )
    # print 'The size of y: '
    # print  y.shape
    return npX, y, Xnames, X, df['y']

def checkIfBetterOff(df):
    X = create_label(df)
    y = df['y']
    return X,y

def updateData(update, df):
    if update:
        X,y = checkIfBetterOff(df)
        X.to_csv('Dataset/XTotal.csv', index=False)
        y.to_csv('Dataset/yTotal.csv', index=False, header=True)


main()
