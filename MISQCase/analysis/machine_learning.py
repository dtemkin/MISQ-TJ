#Prediction and evaluation
import pandas as pd
from scipy.stats import norm
import csv
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (classification_report, precision_score, roc_auc_score, auc, roc_curve,
                             plot_roc_curve, recall_score, f1_score, confusion_matrix)
import numpy as np
import random
from pandas import DataFrame, read_csv
import os
from numpy import array, average
import imblearn

# Read in the data - user_df
def dummy_transform(x):
    if int(x) == 2:
        return 1
    else:
        return 0

def user_val_transform(x):
    if int(x) == 1:
        return 0
    else:
        return 1

def user_level_transform(x):
    if int(x) == 1 or int(x) == -99:
        level = 0
    if int(x) in [2,3]:
        level = 1
    if int(x) in [4,5]:
        level = 2
    else:
        level = 3
    return level

def readData():
    author_df = pd.read_csv(
        f"C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\Data\Analysis Results\MISQ\\person_topics_lemma_19.csv",
        encoding_errors='ignore')
    survey_df = pd.read_csv(
        "C:\\Users\\Tianjie.Deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\Data\Analysis Results\MISQ\\survey_cleaned.csv")
    survey_df = survey_df.rename(
        columns={'Q1.16_8': 'user_id', "Q7.1": "alcohol", "Q7.2": "marijuana", "Q7.3": "cocaine",
                 "Q7.4": "crack",
                 "Q7.5": "heroin", "Q7.6": "meth", "Q7.7": "ecstacy", "Q7.8": "needle", "Q7.9": "prescriptionDrug",
                 "Q1.3": "Age",
                 "Q2.2_1": "genderMale", "Q2.2_2": "genderFemale", "Q2.2_3": "genderTransMale",
                 "Q2.2_4": "genderTransFemale",
                 "Q2.2_5": "genderQueer", "Q2.2_6": "genderOther", "Q2.2_7": "genderDecline",
                 "Q2.1": "Race", "Q2.3": "SexualOrientation", "Q2.4": "Education",
                 "Q2.5": "AttendingSchool", "Q2.6": "Working", "Q2.7": "EverTravelled", "Q2.8": "Traveller",
                 "Q4.1": "PerceivedHealth",
                 "Q6.1": "hadSex",
                 "Q8.5": "OnlineTime",
                 "Q8.8": "DailySNTime", "Q8.9": "WeeklySNTime",
                 "Q8.11_1": "SR_Network", "Q8.11_3": "SR_School",
                 "Q8.11_4": "SR_News", "Q8.11_5": "SR_Knowledge",
                 "Q8.11_11": "SR_Messaging", "Q8.11_7": "SR_MeetPeople", "Q8.11_8": "SR_Sex",
                 "Q8.11_9": "SR_Entertainment", "Q8.11_16": "SR_Family", "Q8.11_17": "SR_Friends",
                 "Q8.11_18": "SR_FillTime", "Q8.11_19": "SR_Politics", "Q8.11_10": "SR_Other",
                 "Q8.11_10_TEXT": "SR_OtherText",
                 "Q8.12": "OnlineEasy",
                 "Q8.13.0": "onlineSexPartner",
                 "Q8.14": "findSexOnline",
                 "Q9.8": "Lonely1", "Q9.9": "Lonely2", "Q9.10": "Lonely3",
                 "Q9.14": "Jail",
                 "Q9.15_4": "Police_alcohol",
                 "Q9.15_5": "Police_panhandling",
                 "Q9.15_6": "Police_sleeping",
                 "Q9.15_7": "Police_gang",
                 "Q9.15_3": "Police_mari",
                 "Q9.15_2": "Police_substance",
                 "Q9.15_8": "Police_other",
                 "Q10.3": "BeAttacked", "Q10.4": "BeingThreathen", "Q10.5": "OthersAttacked",
                 "Q9.5_1": "depression1", "Q9.5_2": "depression2", "Q9.5_3": "depression3",
                 "Q9.5_4": "depression4", "Q9.5_5": "depression5", "Q9.5_6": "depression6",
                 "Q9.5_7": "depression7", "Q9.5_8": "depression8", "Q9.5_9": "depression9",
                 "Q9.3_1": "anxiety1", "Q9.3_2": "anxiety2", "Q9.3_3": "anxiety3",
                 "Q9.3_4": "anxiety4", "Q9.3_5": "anxiety5", "Q9.3_6": "anxiety6",
                 "Q9.3_7": "anxiety7"})
    # Deleted the one with multiple id
    survey_df = survey_df.loc[
        [i for i in survey_df.index if (survey_df.loc[i]['user_id'] != " " and survey_df.loc[i]['user_id'] != np.nan
                                        and survey_df.loc[i]['user_id'] != '149437232298631'
                                        and survey_df.loc[i]['user_id'] != '1979000205677699')]]
    print(f"After deleting duplicated IDs, there are {(len(survey_df))} respondents.")
    survey_df['genderMale'] = survey_df['genderMale'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['genderFemale']=survey_df['genderFemale'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['genderTransMale']=survey_df['genderTransMale'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['genderTransFemale']=survey_df['genderTransFemale'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['genderQueer']=survey_df['genderQueer'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['genderOther']=survey_df['genderOther'].apply(lambda x: 0 if x == ' ' else x)
    survey_df['jail'] = survey_df['Jail'].apply(dummy_transform)
    survey_df['attendingShool'] = survey_df['AttendingSchool'].apply(dummy_transform)
    survey_df['working'] = survey_df['Working'].apply(dummy_transform)
    survey_df['everTravelled'] = survey_df['EverTravelled'].apply(dummy_transform)
    survey_df['beAttacked_dummy'] = survey_df['beAttacked'].apply(dummy_transform)

    survey_df.drop(['AttendingSchool', 'Working','EverTravelled','beAttacked'], axis=1)

    race_dummies_df = pd.get_dummies(survey_df['Race'], prefix='race')
    #education_dummies_df = pd.get_dummies(survey_df['Education'])

    survey_df_new = survey_df.join(race_dummies_df)
    survey_df_new.drop(['Race'], axis=1)
    survey_df_new.to_csv(
        "C:\\Users\\tianjie.deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\SurveyCleaned.csv")

    user_df = author_df.merge(survey_df_new, how='inner', left_on="person_id", right_on="user_id")

    #print(f"The length of the user df after merging is {len(user_df)}")
    # convert binary
    user_df['marijuana_user'] = user_df['marijuana'].apply(user_val_transform)
    user_df['alcohol_user'] = user_df['alcohol'].apply(user_val_transform)
    user_df['cocaine_user'] = user_df['cocaine'].apply(user_val_transform)
    user_df['crack_user'] = user_df['crack'].apply(user_val_transform)
    user_df['heroin_user'] = user_df['heroin'].apply(user_val_transform)
    user_df['meth_user'] = user_df['meth'].apply(user_val_transform)
    user_df['ecstacy_user'] = user_df['ecstacy'].apply(user_val_transform)
    user_df['needle_user'] = user_df['needle'].apply(user_val_transform)
    user_df['prescriptionDrug_user'] = user_df['prescriptionDrug'].apply(user_val_transform)
    # convert overall (0 - does not use drug, 1 - use drug)
    user_df['user'] = user_df[['marijuana_user','cocaine_user','crack_user','heroin_user','meth_user','ecstacy_user',
                               'needle_user','prescriptionDrug_user']].max(axis=1)
    # convert level
    user_df['marijuana_level'] = user_df['marijuana'].apply(user_level_transform)
    user_df['alcohol_level'] = user_df['alcohol'].apply(user_level_transform)
    user_df['cocaine_level'] = user_df['cocaine'].apply(user_level_transform)
    # Because crack only has three values, so just copy
    user_df['crack_level'] = user_df['crack']
    user_df['heroin_level'] = user_df['heroin'].apply(user_level_transform)
    user_df['meth_level'] = user_df['meth'].apply(user_level_transform)
    user_df['ecstacy_level'] = user_df['ecstacy'].apply(user_level_transform)
    user_df['needle_level'] = user_df['needle'].apply(user_level_transform)
    user_df['prescriptionDrug_level'] = user_df['prescriptionDrug'].apply(user_level_transform)


    print("The distribution of marijuana_user user is", pd.DataFrame(user_df.marijuana_user.value_counts()))
    print("The distribution of alcohol user is", pd.DataFrame(user_df.alcohol_user.value_counts()))
    print("The distribution of cocaine user is", pd.DataFrame(user_df.cocaine_user.value_counts()))
    print("The distribution of crack user is", pd.DataFrame(user_df.crack_user.value_counts()))
    print("The distribution of heroin user is", pd.DataFrame(user_df.heroin_user.value_counts()))
    print("The distribution of meth user is", pd.DataFrame(user_df.meth_user.value_counts()))
    print("The distribution of ecstacy user is", pd.DataFrame(user_df.ecstacy_user.value_counts()))
    print("The distribution of needle user is", pd.DataFrame(user_df.needle_user.value_counts()))
    print("The distribution of prescriptionDrug user is", pd.DataFrame(user_df.prescriptionDrug_user.value_counts()))


    user_df['SR_Knowledge'] = [(int(i) if i != " " else 0) for i in user_df['SR_Knowledge']]
    user_df['SR_Network'] = [(int(i) if i != " " else 0) for i in user_df['SR_Network']]
    user_df['SR_School'] = [(int(i) if i != " " else 0) for i in user_df['SR_School']]
    user_df['SR_News'] = [(int(i) if i != " " else 0) for i in user_df['SR_News']]
    user_df['SR_Messaging'] = [(int(i) if i != " " else 0) for i in user_df['SR_Messaging']]
    user_df['SR_MeetPeople'] = [(int(i) if i != " " else 0) for i in user_df['SR_MeetPeople']]
    user_df['SR_Sex'] = [(int(i) if i != " " else 0) for i in user_df['SR_Sex']]
    user_df['SR_Entertainment'] = [(int(i) if i != " " else 0) for i in user_df['SR_Entertainment']]
    user_df['SR_Family'] = [(int(i) if i != " " else 0) for i in user_df['SR_Family']]
    user_df['SR_Friends'] = [(int(i) if i != " " else 0) for i in user_df['SR_Friends']]
    user_df = user_df.drop(['all_posts'], axis=1)
    user_df = user_df.drop(['all_comments'],axis=1)
    user_df['averge_comment'] = user_df['comment_number']/user_df['post_number']
    user_df["like"] = user_df['mean_like']*user_df['post_number']
    user_df["love"] = user_df['mean_love'] * user_df['post_number']
    user_df["wow"] = user_df['mean_wow'] * user_df['post_number']
    user_df["haha"] = user_df['mean_haha'] * user_df['post_number']
    user_df["sad"] = user_df['mean_sad'] * user_df['post_number']
    user_df["angry"] = user_df['mean_angry'] * user_df['post_number']
    user_df["thankful"] = user_df['mean_thankful'] * user_df['post_number']
    user_df["pride"] = user_df['mean_pride'] * user_df['post_number']
    return user_df
#@ignore_warnings(category=ConvergenceWarning)
def evaluation(model,X_test_data,y_test,y_pred):
    accuracy = model.score(X_test_data, y_test)
    rpt = classification_report(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average='macro')
    recall = recall_score(y_test, y_pred,average='macro')
    fs = f1_score(y_test, y_pred, average='macro')

    #fpr, tpr, _ = roc_curve(y_test, y_pred)
    #auc = roc_auc_score(y_test, y_pred)
    try:
        roc_auc_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        auc =0
    fpr = calculate_fpr(list(y_test), list(y_pred))
    return auc, accuracy, precision, recall,fs,fpr

def calculate_fpr(lst_true, lst_pred):
    tn, fp, fn, tp = confusion_matrix(lst_true, lst_pred, labels=[0, 1]).ravel()
    fpr = fp / len(lst_true)
    return fpr

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se
    p = (1 - norm.cdf(abs(t))) * 2
    return p

def build_classifier(X_train, y_train, X_test, y_test,men_xtest, men_ytest, women_xtest,women_ytest,
                     other_xtest, other_ytest,young_xtest, young_ytest,old_xtest,old_ytest,
                     baseclf):

    baseclf.fit(X_train, y_train)

    #importance
    #model_coeff = np.mean([ns.coef_ for ns in baseclf.estimators_], axis=0)
    # Multiply the model coefficients by the standard deviation of the data
    #coeff_magnitude = np.std(X_train, 0) * model_coeff

    #p-values
    #pValues = np.mean([logit_pvalue(ns, X_train) for ns in baseclf.estimators_], axis=0)

    pred = baseclf.predict(X_test)
    actual = y_test
    auc, accuracy, precision, recall, fs, fpr = evaluation(baseclf, X_test, actual, pred)

    # men and women
    men_pred = baseclf.predict(men_xtest)
    women_pred = baseclf.predict(women_xtest)
    if(len(other_xtest)==0):
        auc_ot, accuracy_ot, precision_ot, recall_ot, fs_ot, fpr_ot = 0,0,0,0,0,0
    else:
        other_pred = baseclf.predict(other_xtest)
        auc_ot, accuracy_ot, precision_ot, recall_ot, fs_ot, fpr_ot = evaluation(baseclf, other_xtest, other_ytest,
                                                                                 other_pred)


    auc_m, accuracy_m, precision_m, recall_m, fs_m, fpr_m = evaluation(baseclf, men_xtest, men_ytest, men_pred)
    auc_w, accuracy_w, precision_w, recall_w, fs_w, fpr_w = evaluation(baseclf, women_xtest, women_ytest, women_pred)

    # yound and old
    young_pred = baseclf.predict(young_xtest)
    old_pred = baseclf.predict(old_xtest)
    auc_y, accuracy_y, precision_y, recall_y, fs_y, fpr_y = evaluation(baseclf, young_xtest, young_ytest, young_pred)
    auc_o, accuracy_o, precision_o, recall_o, fs_o, fpr_o = evaluation(baseclf, old_xtest, old_ytest, old_pred)


    return auc, accuracy, precision, recall,fs, fpr, \
           auc_m, accuracy_m, precision_m, recall_m, fs_m,fpr_m, \
           auc_w, accuracy_w, precision_w, recall_w, fpr_w, \
           auc_ot, accuracy_ot, precision_ot, recall_ot, fs_ot, fpr_ot,\
           fs_w,auc_y, accuracy_y, precision_y, recall_y, fs_y, fpr_y, \
           auc_o, accuracy_o, precision_o, recall_o, fs_o, fpr_o



def prep_data(user_df, X_feats, y_feat,test_size=0.3):
    #Read the user data
    X = user_df[X_feats].to_numpy()
    y0 = user_df[["genderMale", 'Age', 'genderFemale', y_feat]].to_numpy()
    y1 = user_df[y_feat].to_numpy()
    print("the distribution in the y is like", dict(Counter(y1)))
    # split train_test

    #X_train, X_test, y_train, y_test = train_test_split(X, y0, random_state=120, test_size=test_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y0, random_state=120, test_size=test_size, stratify=y1)

    print("the distribution in the y test is like", dict(Counter(y_test[:,3])))



    men_idx, female_idx = [], []
    other_idx = []
    young_idx, old_idx = [], []
    for i in range(len(y_test)):

        if y_test[i][0] == "1":
            men_idx.append(i)
        elif y_test[i][2] == "1":
            female_idx.append(i)
        else:
            other_idx.append(i)
        if int(y_test[i][1]) in [1,2,3,4]:
            young_idx.append(i)
        elif int(y_test[i][1]) in [5,6,7,8]:
            old_idx.append(i)

    y_train = y_train[:, 3]
    y_train = y_train.astype('int')


    men_xtest = X_test[men_idx, :]
    men_ytest = y_test[men_idx, 3]
    men_ytest = men_ytest.astype('int')
    women_xtest = X_test[female_idx, :]
    women_ytest = y_test[female_idx, 3]
    women_ytest = women_ytest.astype('int')
    other_xtest = X_test[other_idx, :]
    other_ytest = y_test[other_idx, 3]
    other_ytest = other_ytest.astype('int')
    young_xtest = X_test[young_idx, :]
    young_ytest = y_test[young_idx, 3]
    young_ytest = young_ytest.astype('int')
    old_xtest = X_test[old_idx, :]
    old_ytest = y_test[old_idx, 3]
    old_ytest = old_ytest.astype('int')

    y_test = y_test[:, 3]
    y_test = y_test.astype('int')
    print("number of men", len(men_idx))
    print("number of woman", len(female_idx))
    print("number of other", len(other_idx))
    print("number of young", len(young_idx))
    print("number of old", len(old_idx))

    return X_train, X_test, y_train, y_test, men_xtest, men_ytest, women_xtest,women_ytest, \
           other_xtest,other_ytest,young_xtest, young_ytest, old_xtest,old_ytest


def feature_prep(optimal_number):
    #topicList = [f"topic{i}" for i in range(optimal_number) if i not in [5,6,8,9,10,11]]
    topicList = [f"topic{i}" for i in range(optimal_number)]
    post_info = ["post_number",'mean_post_length']
    post_sentiment = ["post_sentiment"]
    #post_sentiment = ["post_positive","post_negative","post_neautral"]
    post_emotion = ['post_happy','post_angry','post_surprise','post_sad','post_fear']
    reaction = ['mean_like', 'mean_love','mean_wow','mean_haha','mean_sad','mean_angry']
    reaction2 = ["like","love","wow","haha","sad","angry"]
    comment = ["comment_sentiment","comment_number"]
    #comment =["comment_positive","comment_negative","comment_neautral","comment_number"]
    comment_emotion = ["comment_happy", "comment_angry","comment_surprise",'comment_sad','comment_fear']
    drug_use = ['alcohol_user','prescriptionDrug_user','cocaine_user','crack_user','heroin_user','meth_user','ecstacy_user','needle_user']
    #survey_info = ['Age','PerceivedHealth','WeeklySNTime',"SR_Network", "SR_School","SR_News", "SR_Knowledge",
                                        #"SR_Messaging",  "SR_MeetPeople",  "SR_Sex",
                                        #"SR_Entertainment", "SR_Family", "SR_Friends",
                   #'working','attendingShool','jail','everTravelled','beAttacked_dummy','Education',
                   #"race_1","race_2","race_3","race_4","race_5","race_5",
                   #"genderMale"]

    survey_info = ['Age', 'PerceivedHealth', 'working', 'attendingShool', 'jail', 'everTravelled', 'beAttacked_dummy', 'Education',
                  "race_1", "race_2", "race_3", "race_4", "race_5", "race_5",
                  "genderMale",'depression1','anxiety1']

    x_feature_sets = {"feature2": post_info + post_sentiment + reaction + comment + topicList,
                      "feature0": post_info + reaction + comment+topicList,
                      "feature1": post_info +reaction+comment,
                      "feature3": post_info + reaction,
                      "feature4": post_info,
                      #"feature4": post_info+post_sentiment+reaction,
                     'feature5':  post_info+post_emotion+reaction+comment+topicList+comment_emotion,
                     "feature6": post_info+post_sentiment + reaction + comment + topicList+survey_info,
                      "feature7":survey_info
                      }
    return x_feature_sets

def run(user_df, optimal_number, estimator, estimator_name,feature_set="feature1",target="meth_user"):
    x_feature_sets = feature_prep(optimal_number)
    x_feature_set = x_feature_sets[feature_set]
    print("features are", x_feature_set)
    X_train, X_test, y_train, y_test, men_xtest, men_ytest, women_xtest, women_ytest, other_xtest,other_ytest, young_xtest, young_ytest, old_xtest, old_ytest = prep_data(user_df, X_feats=x_feature_set, y_feat=target, test_size=0.35)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    clf_perfs = []

    feature_importanceLst = []
    for i in range(optimal_number):


        auc, accuracy, precision, recall,fs, fpr,\
        auc_m, accuracy_m, precision_m, recall_m,fs_m,fpr_m,\
        auc_w, accuracy_w, precision_w, recall_w,fs_w,fpr_w, \
        auc_ot, accuracy_ot, precision_ot, recall_ot, fs_ot, fpr_ot,\
        auc_y, accuracy_y, precision_y, recall_y,fs_y,fpr_y,\
        auc_o, accuracy_o, precision_o, recall_o, fs_o, fpr_o = build_classifier(X_train, y_train, X_test, y_test, men_xtest, men_ytest,
                                                                        women_xtest,women_ytest,
                                                                        other_xtest, other_ytest,
                                                                        young_xtest, young_ytest,
                                                                        old_xtest, old_ytest,
                                                                        baseclf=estimator)
        perf_dict = {'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, "f1": fs, 'fpr': fpr,
                     'auc_m': auc_m, 'accuracy_m': accuracy_m, 'precision_m': precision_m, 'recall_m': recall_m, "f1_m": fs_m, 'fpr_m': fpr_m,
                     'auc_w': auc_w, 'accuracy_w': accuracy_w, 'precision_w': precision_w, 'recall_w': recall_w, "f1_w": fs_w, 'fpr_w': fpr_w,
                     'auc_ot': auc_ot, 'accuracy_ot': accuracy_ot, 'precision_ot': precision_ot, 'recall_ot': recall_ot, "f1_ot": fs_ot, 'fpr_ot': fpr_ot,
                     'auc_y': auc_y, 'accuracy_y': accuracy_y, 'precision_y': precision_y, 'recall_y': recall_y, "f1_y": fs_y, 'fpr_y': fpr_y,
                     'auc_o': auc_o, 'accuracy_o': accuracy_o, 'precision_o': precision_o, 'recall_o': recall_o, "f1_o": fs_o, 'fpr_o': fpr_o}
        # performance of our models

        clf_perfs.append(perf_dict)

        #plt.plot(fpr, tpr)
        #plt.ylabel('True Positive Rate')
        #plt.xlabel('False Positive Rate')
        #plt.show()
        #feature_importanceLst.append(feature_importances)
        #featureLen = len(feature_importances)
    # calculate the average of feature importance
    #array_feature = array(feature_importanceLst)

    #np.hsplit(array_feature,featureLen)
    #performance_average = average(array_feature, axis=0)
    # making the performance plot and save the performance matrics
    performance_df = pd.DataFrame.from_records(clf_perfs, columns=list(clf_perfs[0].keys()))
    #performance_df.hist('auc')
    #plt.show()
    #performance_df.hist("accuracy")
    #plt.show()
    #print(performance_average)
    performance_df.to_csv(f"C:\\Users\\tianjie.deng\\Dropbox\\"
                          f"PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\performance_{optimal_number}clusters_{estimator_name}_{feature_set}_fullTopics_{target}.csv")


if __name__=="__main__":
        user_df = readData()
        user_df.to_csv("C:\\Users\\tianjie.deng\\Dropbox\\PROF PREP\\DCB Facebook\\Facebook Data Analysis\\Data\\Analysis Results\\MISQ\\FullData_19Topics.csv")


        #for feature_set in ["feature0", "feature1","feature2","feature3",'feature4','feature5','feature6']:
        for feature_set in ["feature6","feature7"]:
            for target in ['marijuana_user']:
            #for target in ["marijuana_user",'alcohol_user','cocaine_user','crack_user','heroin_user','meth_user', 'needle_user', 'prescriptionDrug_user','ecstacy_user']:
                print(f"analyzing {target}")
                common_params = {"n_estimators": 1000, "max_samples":0.8, "bootstrap":True}
                 #models = {"log1_bagged": BaggingClassifier(LogisticRegression(solver='newton-cg', class_weight='balanced'),
                                                       #**common_params),
                      #'decision_tree': BaggingClassifier(DecisionTreeClassifier(class_weight='balanced'),**common_params),
                      #'svc_bagged': BaggingClassifier(SVC(), **common_params),
                      #'random_forest': RandomForestClassifier(max_features=0.8,class_weight='balanced', **common_params),
                    #'ada_boost': AdaBoostClassifier()}
                #solver used to use newton-cg but failed to converge liblinear
                models = {"log1_bagged": BaggingClassifier(LogisticRegression(solver='newton-cg', class_weight='balanced',
                        multi_class='multinomial'), **common_params)}

                for mod_name in models:
                    run(user_df, optimal_number = 19, feature_set=feature_set, estimator=models[mod_name],
                        estimator_name=mod_name,target=target)
                print(f"Finishing analyzing {target}")
                print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            #run(19, feature_set=feature_set, estimator='svc')
            #run(19, feature_set=feature_set, estimator='log')
            #run(19,feature_set=feature_set)
