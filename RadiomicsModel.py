from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import preprocessing
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.utils import resample
from nonconformist.cp import IcpClassifier
from nonconformist.nc import NcFactory
import seaborn as sn


def GenerateCI(model, X_train, Y_train, X_val, Y_val):
    """
    Generate the confidence intervall on different classifier
    :param model:
    :param X_train:
    :param Y_train:
    :param X_val:
    :param Y_val:
    :return:
    """
    SAM = int(len(X_val.index) * 0.9)
    score = []
    for i in range(100):
        X, y = resample(X_val, Y_val, n_samples=SAM, random_state=i, stratify=Y_val)
        if model == "RF":
            res = RandomForestClass(X_train, Y_train, X, y, False)
        elif model == "SVM":
            res = SvmModel(X_train, Y_train, X, y, False)
        else:
            res = LogisticClass(X_train, Y_train, X, y, False)
        score.append(res)

    print(f"Confidence inteval for CI: {list(confidence_interval(score))[1:]}" + str(model))
    plt.hist(score, bins=10)
    plt.show()


def trimm_correlated(df_in, threshold):
    """
    Perform correlatio nanalysis
    :param df_in:
    :param threshold:
    :return:
    """
    df_corr = df_in.corr(method='spearman', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    # df_out = df_in[un_corr_idx]
    return un_corr_idx


def returnrandom():
    """
    Help function for the estimator of the random forest setting
    :return:
    """
    n_estimators = [int(x) for x in np.linspace(start=100, stop=600, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 40, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    return random_grid




def plot_precis(precision, recall):
    """
    :param recall:
    :param precision:
    :return:
    """
    plt.plot(recall, precision, color='blue', label='ROC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()


def transform_label(csv, col_name, type1, type2):
    csv[col_name] = csv[col_name].replace([type1], type2)
    return csv


def Wilcox(X, Y):
    rd = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=1)
    Svm = svm.SVC(kernel='linear', class_weight='balanced', probability=True)
    kf = StratifiedKFold(n_splits=20)
    results_model1 = cross_val_score(rd, X, Y, cv=kf, scoring='roc_auc')
    results_model2 = cross_val_score(Svm, X, Y, cv=kf, scoring='roc_auc')
    stat, p = wilcoxon(results_model1, results_model2, zero_method='zsplit')
    return p


def split_single_all_in3(origin):
    # dt = {'StudySubjectID': 'str', 'Center': 'int', 'Pneumonitis': 'int'}

    # train = origin[origin['Center'].isin(list_train)]
    # train = train.drop("Center", axis=1)
    st_sub = origin.drop('StudySubjectID', axis=1)
    st_sub = st_sub.drop('Center', axis=1)
    st_sub = st_sub.drop('KVP', axis=1)
    st_sub = st_sub.drop('Manufacturer', axis=1)
    label = origin['Pneumonitis']

    X_train, X_val, y_train, y_val = train_test_split(
        st_sub, pd.concat([st_sub[["SliceThinkness"]], label], axis=1), test_size=0.2,
        stratify=pd.concat([st_sub[["SliceThinkness"]], label], axis=1), random_state=1)
    label = X_train['Pneumonitis']

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, pd.concat([X_train[["SliceThinkness"]], label], axis=1), test_size=0.1,
        stratify=pd.concat([X_train[["SliceThinkness"]], label], axis=1), random_state=1)
    X_train = X_train.drop(['SliceThinkness'], axis=1)
    X_val = X_val.drop(['SliceThinkness'], axis=1)
    X_test = X_test.drop(['SliceThinkness'], axis=1)

    return X_train, X_val, X_test


def split(origin, list_test, list_train):
    # origin = origin.drop('Manufacturer', axis=1)
    origin = origin.drop('StudySubjectID', axis=1)
    # origin = origin.drop('KVP', axis=1)
    # origin = origin.drop('SliceThinkness', axis=1)
    # origin = origin.drop('Manufacturer', axis=1)
    test = origin[origin['Center'].isin(list_test)]
    train = origin[origin['Center'].isin(list_train)]
    test = test.drop("Center", axis=1)
    train = train.drop("Center", axis=1)

    return train, test


def plot_roc_curve(fpr, tpr, title, dire):
    """
    :param fpr: False positive rate
    :param tpr: True positive rate
    :return:
    """
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve' + str(title))
    plt.legend()
    # plt.savefig(dire + title + "AUC.png")
    plt.show()


def GetNumberOfPcaComp(data):
    """
    Plot the number of PCA compent per variance explained
    :param data:
    :return:
    """
    pca = PCA().fit(data)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


def ROC_and_Cali_Plot(true_value, pred_prob, title, dire):
    """
    plot calibration and AUC ROC score
    :param true_value:
    :param pred_prob:
    :param title:
    :param dire:
    :return:
    """
    print(roc_auc_score(true_value, pred_prob))
    fpr, tpr, thresholds = roc_curve(true_value, pred_prob, pos_label=1)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    thresholded_pred = np.where(pred_prob >= thresholds[ix], 1, 0)
    print(classification_report(true_value, thresholded_pred))

    v_true_int = np.array(true_value)
    print(v_true_int.astype(int))
    data = {'true': v_true_int.astype(int), "predicted": thresholded_pred}
    df = pd.DataFrame(data=data)
    confusion_m = pd.crosstab(df["true"], df["predicted"], rownames=["true"], colnames=["predicted"])
    sn.heatmap(confusion_m, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.show()

    plot_roc_curve(fpr, tpr, title, dire)

    # reliability diagram
    dt = {"true_value": true_value, "Predict_prob": pred_prob}
    file = pd.DataFrame(data=dt)
    # file.to_excel("D:/prob.xls", index=False)
    # file.to_csv("D:/probcsv.csv", index=False)
    fop, mpv = calibration_curve(true_value, pred_prob, n_bins=10)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(mpv, fop, marker='.')
    plt.title(title)
    # plt.savefig(dire + title + "Cali.png")
    plt.show()


def SvmModel(X_t, Y_T, X_v, Y_v, dire, plot=False, ):
    """
        :param X_t: train x
        :param Y_T: label x
        :param X_v: val x
        :param Y_v:  val y
        :param dire: save direct
        :param plot: boolean
        :return:
        """
    clf = svm.SVC(class_weight='balanced', probability=True)
    clf = clf.fit(X_t, Y_T)
    y_pred = clf.predict(X_v)
    y_pred_pt = clf.predict_proba(X_v)[:, 1]
    if plot:
        print("Auc score SVG:", roc_auc_score(Y_v, clf.predict_proba(X_v)[:, 1]))
        ROC_and_Cali_Plot(Y_v, y_pred_pt, "SVM", dire)

        average_precision = average_precision_score(Y_v, y_pred_pt, average='weighted')
        print("average preci:", average_precision)
        disp = plot_precision_recall_curve(clf, X_v, Y_v, pos_label=1)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))
        plt.show()
    return roc_auc_score(Y_v, clf.predict_proba(X_v)[:, 1])



def CenterSplit(data):
    """
    Split by center of acquisiton
    :param data:
    :return:
    """
    list_t = [2, 3, 4, 5, 6]
    list_v = [1]
    x_train, x_val = split(data, list_v, list_t)
    y_train = x_train['Pneumonitis']
    y_val = x_val['Pneumonitis']
    x_train = x_train.drop('Pneumonitis', axis=1)
    x_val = x_val.drop('Pneumonitis', axis=1)
    return x_train, y_train, x_val, y_val


def EstimateBestRF(X_t, Y_T, X_v, Y_v):
    """
    Estimate parameter random forest
    :param X_t:
    :param Y_T:
    :param X_v:
    :param Y_v:
    :return:
    """
    random_grid = returnrandom()

    rf = RandomForestClassifier(class_weight='balanced')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                   random_state=42, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(X_t, Y_T)
    print(rf_random.best_params_)


def RandomForestClass(X_t, Y_T, X_v, Y_v, dire, plot=False):
    """
        :param X_t: train x
        :param Y_T: label x
        :param X_v: val x
        :param Y_v:  val y
        :param dire: save direct
        :param plot: boolean
        :return:
        """
    clf = RandomForestClassifier(class_weight='balanced', n_estimators=400, min_samples_split=4, min_samples_leaf=1,
                                 max_features='sqrt', max_depth=None, bootstrap=False, random_state=1)
    clf = clf.fit(X_t, Y_T)
    y_pred = clf.predict(X_v)
    y_pred_pt = clf.predict_proba(X_v)[:, 1]

    # print(classification_report(Y_v, y_pred))
    if plot:
        print("Auc  RF: ", roc_auc_score(Y_v, y_pred_pt))
        ROC_and_Cali_Plot(Y_v, y_pred_pt, 'RandomForest', dire)
        average_precision = average_precision_score(Y_v, y_pred_pt, )
        print("Average precision: ", average_precision)
        disp = plot_precision_recall_curve(clf, X_v, Y_v, pos_label=1)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))
        plt.show()
    return roc_auc_score(Y_v, y_pred_pt)


def LogisticClass(X_tt, Y_T, X_vv, Y_vv, dire, plot=False):
    """

    :param X_tt: train x
    :param Y_T: label x
    :param X_vv: val x
    :param Y_vv:  val y
    :param dire: save direct
    :param plot: boolean
    :return:
    """
    clf = LogisticRegression(class_weight='balanced', solver="liblinear", penalty='l1', random_state=2)

    # clf = CalibratedClassifierCV(clf, method='isotonic', cv=10)
    clf.fit(X_tt, Y_T)
    y_pred = clf.predict(X_vv)
    y_pred_pt = clf.predict_proba(X_vv)[:, 1]
    y_pred_x = clf.predict_proba(X_tt)[:, 1]

    # print(classification_report(Y_v, y_pred))
    if plot:
        print("Auc  Logistic: ", roc_auc_score(Y_vv, y_pred_pt))
        ROC_and_Cali_Plot(Y_vv, y_pred_pt, 'Features Level', dire)
        average_precision = average_precision_score(Y_vv, y_pred_pt, )
        print("Average precision: ", average_precision)
        disp = plot_precision_recall_curve(clf, X_vv, Y_vv, pos_label=1)
        disp.ax_.set_title('2-class Precision-Recall curve: '
                           'AP={0:0.2f}'.format(average_precision))
        plt.show()
    return roc_auc_score(Y_vv, y_pred_pt)


def MinMaxScale(train, val, test=None):
    """
    scale normalized featueres
    :param train:
    :param val:
    :param test:
    :return:
    """
    min_max_scaler = preprocessing.MinMaxScaler()
    df_s = min_max_scaler.fit_transform(train)
    train_s = pd.DataFrame(df_s, columns=train.columns, index=train.index)

    df_s = min_max_scaler.transform(val)
    val_s = pd.DataFrame(df_s, columns=val.columns, index=val.index)
    if test is not None:
        df_s = min_max_scaler.transform(test)
        test_s = pd.DataFrame(df_s, columns=test.columns, index=test.index)
        return train_s, val_s, test_s
    else:
        return train_s, val_s


def RFE_(x, y, feature_name):
    """
    Recursive feature elimination
    :param x: x value
    :param y: target value
    :param feature_name: columns name
    :return:
    """
    lr = LogisticRegression(C=.01, solver='liblinear')
    rfe = RFE(lr)
    rfe.fit(x, y)

    feat_sel = []
    for x1, y1 in (sorted(zip(rfe.ranking_, feature_name), key=itemgetter(0))):

        if x1 == 1 and y1 != 'original_shape_MeshVolume':
            feat_sel.append(y1)

    return feat_sel


def confidence_interval(stats, alpha=0.95):
    """
    Generate confidence itnervall
    :param stats:
    :param alpha:
    :return:
    """
    # confidence intervals
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    return np.mean(stats), lower, upper


def save(csv1, csv2):
    """

    :param csv1:
    :param csv2:
    :return:
    """
    dt = {'StudySubjectID': 'int'}
    csv1 = pd.read_csv(csv1, sep=',', dtype=dt)
    csv2 = pd.read_csv(csv2, sep=',')
    list_features_lung = ['original_glcm_DifferenceVariance', 'original_glcm_Idmn',
                          'original_glszm_GrayLevelNonUniformityNormalized',
                          'original_glszm_LowGrayLevelZoneEmphasis',
                          'wavelet-LHL_glszm_ZoneEntropy', 'wavelet-HLH_glcm_ClusterShade',
                          'wavelet-HHH_glszm_SizeZoneNonUniformityNormalized',
                          'wavelet-LLL_glszm_LowGrayLevelZoneEmphasis', "StudySubjectID"]
    csv1 = csv1[list_features_lung]
    new = pd.merge(csv1, csv2, on=['StudySubjectID'], how='left')
    new.to_csv("D:/radForComb.csv")


def preprocess(csv_radiomics, csv_label, how_split='center', do_pca=False):
    """

    :param csv_radiomics: radiomics value csv
    :param csv_label: csv with the label for patient
    :param how_split: type of split
    :param do_pca: apply PCA
    :return:
    """
    dt = {'StudySubjectID': 'int'}
    data = pd.read_csv(csv_radiomics, sep=',', dtype=dt)
    print(data.head())
    print(len(data.columns))
    # and 'firstorder' not in colum and 'glrlm' not in colum

    for colum in data.columns:  # Select the group of features to use
        if 'glcm' not in colum and 'glszm' not in colum \
                and colum != 'StudySubjectID' not in colum:
            data = data.drop(colum, axis=1)  # Drop non informative columns

    label_data = pd.read_csv(csv_label, sep=',')
    print(label_data.head())
    new = pd.merge(data, label_data, on=['StudySubjectID'], how='left')  # Merge with labels
    print('Initian number of features :', len(new.columns))
    print("before drop na", len(new.index))
    new = new.dropna()
    print("after drop na ", len(new.index))

    label = new['Pneumonitis']

    if how_split == 'center':  # Split based on center
        x_train, y_train, x_val_s, y_val = CenterSplit(new)
    else:
        x_train, x_val_s, y_train, y_val = train_test_split(new, label, test_size=0.2, stratify=label, random_state=1)

    x_train_s, x_val_s = MinMaxScale(x_train, x_val_s)  # Normalization

    features = RFE_(x_train_s, y_train, x_train_s.columns)  # Recursive Features Selection

    x_train_s = x_train_s[features]
    x_val_s = x_val_s[features]

    print("Number of Features after RFE ", len(x_train_s.columns))
    low_corr = trimm_correlated(x_train_s, 0.60)  # Drop High Correlated
    print(low_corr)

    x_train_s = x_train_s[low_corr]
    x_val_s = x_val_s[low_corr]

    print("Number of Features after Correlation analysis ", len(x_train_s.columns))
    if do_pca:  # Apply PCA or not
        GetNumberOfPcaComp(x_train_s)
        pca = PCA(n_components=10)
        x_train_s = pca.fit_transform(x_train_s)
        x_val_s = pca.transform(x_val_s)
        c_n = []
        for i in range(10):
            c_n.append('component' + str(i))

        x_train_s = pd.DataFrame(data=x_train_s
                                 , columns=c_n)
        x_val_s = pd.DataFrame(data=x_val_s
                               , columns=c_n)

    return x_train_s, y_train, x_val_s, y_val


def CompareModel(X_train, Y_train, X_val, Y_val, dire, Multi=False):
    if not Multi:
        GenerateCI("RF", X_train, Y_train, X_val, Y_val)
        GenerateCI("SVM", X_train, Y_train, X_val, Y_val)
        GenerateCI("log", X_train, Y_train, X_val, Y_val)
        # GenerateCI("log", X_train, Y_train, X_val, Y_val)
        LogisticClass(X_train, Y_train, X_val, Y_val, dire, plot=True)
        RandomForestClass(X_train, Y_train, X_val, Y_val, dire, plot=True)
        SvmModel(X_train, Y_train, X_val, Y_val, dire, plot=True)
    else:
        print("multi class")
        RandomForestClassMulti(X_train, Y_train, X_val, Y_val)


if __name__ == '__main__':
    X_t, Y_t, X_v, Y_v = preprocess("D:/RadiomicsSingleLungALLp.csv", "D:/Bms_csv/IorNotTotal.csv",
                                    how_split='center', do_pca=False)
    # RandomForestClassNoAuc(X_t, Y_t, X_v, Y_v)
    # EstimateBestRF(X_t, Y_t, X_v, Y_v)
    CompareModel(X_t, Y_t, X_v, Y_v, 'D:/PlotBms/')

    # save("D:/RadiomicsSingleLungALLp.csv", "D:/Bms_csv/IorNotTotal.csv")
