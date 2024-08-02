import os

import numpy as np
import pandas as pd
import pyreadr
from sklearn.conftest import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler


def is_discrete_array(arr):
    # Check if array data type is integer or all elements are whole numbers
    return np.issubdtype(arr.dtype, np.integer) or np.all(np.mod(arr, 1) == 0)


def load_enhancer_data(data_path, group_func="mean"):
    assert os.path.isfile(data_path)
    assert group_func in ["mean", "max", "first", "random", "pca"]
    r_data_obj = pyreadr.read_r(data_path)
    X = r_data_obj["X"].values
    Y = r_data_obj["Y"].values
    varnames_df = r_data_obj['varnames.all']
    collapsed_feat_names = varnames_df.Predictor_collapsed.unique()
    full_feat_names = varnames_df.Predictor.values
    grouped_X_df = {}
    for feat_name in collapsed_feat_names:
        curr_df = varnames_df[varnames_df.Predictor_collapsed == feat_name]
        full_feat_index = curr_df.index.values
        assert len(full_feat_index) > 0
        curr_X = X[:, full_feat_index]
        if group_func == "mean":
            grouped_X_df[feat_name] = curr_X.mean(axis=1)
        elif group_func == "max":
            grouped_X_df[feat_name] = curr_X.max(axis=1)
        elif group_func == "first":
            grouped_X_df[feat_name] = curr_X[:, 0]
        elif group_func == "random":
            rand_idx = np.random.choice(full_feat_index, size=X.shape[0])
            grouped_X_df[feat_name] = X[np.arange(X.shape[0]), rand_idx]
        elif group_func == "pca":
            if curr_X.shape[1] > 1:
                grouped_X_df[feat_name] = PCA(n_components=1).fit_transform(curr_X).reshape(-1)
            else:
                grouped_X_df[feat_name] = curr_X.reshape(-1)

    grouped_X_df = pd.DataFrame(grouped_X_df)
    X = grouped_X_df.values
    feat_names = grouped_X_df.columns.values

    # shuffle sample indices
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]

    X = MaxAbsScaler().fit_transform(X)
    
    return X, Y, feat_names


def get_drosophila_enhancer_gt():
    return ["gt2", "wt_ZLD", "twi", "hb", "kr", "bcd",
     "med2"],[
        {"gt2", "wt_ZLD"}, {"twi", "wt_ZLD"}, {"gt2", "hb"},
        {"gt2", "kr"}, {"gt2", "twi"}, {"kr", "twi"}, 
        {"bcd", "gt2"}, {"bcd", "twi"}, {"hb", "twi"}, 
        {"med2", "twi"}, {"med2", "wt_ZLD"}, {"hb", "wt_ZLD"}, 
        {"hb", "kr"}, {"bcd", "kr"}, {"bcd", "wt_ZLD"}, {"wt_ZLD", "kr"}
    ]

def inter_gt_name2idx(named_inter_gt, name_list):
    name_list = np.array(name_list)
    inter_gt = []
    name_list = np.array(name_list)
    for (name1, name2) in named_inter_gt:
        idx1 = np.where(name_list==name1)[0]
        idx2 = np.where(name_list==name2)[0]
        assert len(idx1) == 1 and len(idx2) == 1
        inter_gt.append({int(idx1[0]), int(idx2[0])})
    return inter_gt

def import_gt_name2idx(named_import_gt, name_list):
    name_list = np.array(name_list)
    import_gt = []
    for name in named_import_gt:
        idx = np.where(name_list==name)[0]
        assert len(idx) == 1
        import_gt.append(int(idx[0]))
    return import_gt

def sel_inter_idx2name(sel_inter, name_list):
    named_sel_inter = []
    for idx, score in sel_inter:
        i, j = idx
        name1, name2 = name_list[i], name_list[j]
        named_sel_inter.append(((name1, name2), score))
    return named_sel_inter

def sel_import_idx2name(sel_import, name_list):
    return [name_list[i] for i in sel_import]


def load_mortality_data(data_dir):
    '''
    Load NHANES I biochemistry tape and mortality data.
    '''
    biochemtapepath = os.path.join(data_dir, "DU4800.txt") 
    assert os.path.isfile(biochemtapepath)
    medexamtapepath = os.path.join(data_dir, 'DU4233.txt')
    assert os.path.isfile(medexamtapepath)
    anthropometrypath = os.path.join(data_dir, "DU4111.txt")
    assert os.path.isfile(anthropometrypath)
    vitlpath = os.path.join(data_dir, "N92vitl.txt")
    assert os.path.isfile(vitlpath)
    d = {}
    with open(biochemtapepath, 'r') as handle:
        for line in handle:
            seqn = int(line[0:5])
            d[seqn] = {}

            # Date of examination
            d[seqn]['exam_month'] = int(line[137:139])
            d[seqn]['exam_year'] = int(line[141:143])

            # 1 if male, 2 if female
            sex = int(line[103])
            # d[seqn]['sex_isMale'] = (sex == 1)
            d[seqn]['sex_isFemale'] = (sex == 2)

            # Age at examination
            d[seqn]['age'] = int(line[143:145])

            # Physical activity in past 24 hours?
            d[seqn]['physical_activity'] = int(line[225])
            # Is the field 8? (blank)
            d[seqn]['physical_activity_isBlank'] = (d[seqn]['physical_activity'] == 8)

            # Serum albumin
            try:
                d[seqn]['serum_albumin'] = float(line[231:235])
            except ValueError:
                d[seqn]['serum_albumin'] = np.nan
            d[seqn]['serum_albumin_isBlank'] = np.isnan(d[seqn]['serum_albumin'])
            d[seqn]['serum_albumin_isMissingAge1to3'] = (d[seqn]['serum_albumin'] == 9999)
            if d[seqn]['serum_albumin'] == 9999:
                d[seqn]['serum_albumin'] = np.nan
            d[seqn]['serum_albumin'] /= 10
            try:
                isImputed = (int(line[235]) == 1)
            except ValueError:
                isImputed = False
            if isImputed:
                d[seqn]['serum_albumin'] = np.nan
                # I assume that the study having imputed the value is the same 
                # as the study having left the field blank
                d[seqn]['serum_albumin_isBlank'] = True


            # Alkaline phosphatase
            try:
                d[seqn]['alkaline_phosphatase'] = float(line[458:462])
            except ValueError:
                d[seqn]['alkaline_phosphatase'] = np.nan
            d[seqn]['alkaline_phosphatase_isUnacceptable'] = (d[seqn]['alkaline_phosphatase'] == 7777)
            d[seqn]['alkaline_phosphatase_isBlankbutapplicable'] = (d[seqn]['alkaline_phosphatase'] == 8888)
            d[seqn]['alkaline_phosphatase_isTestnotdone'] = (d[seqn]['alkaline_phosphatase'] == 9999)
            d[seqn]['alkaline_phosphatase_isBlank'] = np.isnan(d[seqn]['alkaline_phosphatase'])
            if d[seqn]['alkaline_phosphatase'] in [7777, 8888, 9999]:
                d[seqn]['alkaline_phosphatase'] = np.nan
            d[seqn]['alkaline_phosphatase'] /= 10

            # SGOT/aspartate aminotransferase
            try: 
                d[seqn]['SGOT'] = float(line[454:458])
            except ValueError:
                d[seqn]['SGOT'] = np.nan
            d[seqn]['SGOT_isUnacceptable'] = (d[seqn]['SGOT'] == 7777)
            d[seqn]['SGOT_isBlankbutapplicable'] = (d[seqn]['SGOT'] == 8888)
            d[seqn]['SGOT_isTestnotdone'] = (d[seqn]['SGOT'] == 9999)
            d[seqn]['SGOT_isBlank'] = np.isnan(d[seqn]['SGOT'])
            if d[seqn]['SGOT'] in [7777, 8888, 9999]:
                d[seqn]['SGOT'] = np.nan
            d[seqn]['SGOT'] /= 100

            # BUN (blood urea nitrogen)
            try:
                d[seqn]['BUN'] = float(line[471:474])
            except ValueError:
                d[seqn]['BUN'] = np.nan
            d[seqn]['BUN_isUnacceptable'] = (d[seqn]['BUN'] == 777)
            d[seqn]['BUN_isTestnotdone'] = (d[seqn]['BUN'] == 999)
            d[seqn]['BUN_isBlank'] = np.isnan(d[seqn]['BUN'])
            if d[seqn]['BUN'] in [777, 999]:
                d[seqn]['BUN'] = np.nan
            d[seqn]['BUN'] /= 10

            # Calcium
            try: 
                d[seqn]['calcium'] = float(line[465:468])
            except ValueError:
                d[seqn]['calcium'] = np.nan
            d[seqn]['calcium_isUnacceptable'] = (d[seqn]['calcium'] == 777)
            d[seqn]['calcium_isBlankbutapplicable'] = (d[seqn]['calcium'] == 888)
            d[seqn]['calcium_isTestnotdone'] = (d[seqn]['calcium'] == 999)
            d[seqn]['calcium_isBlank'] = np.isnan(d[seqn]['calcium'])
            if d[seqn]['calcium'] in [777, 888, 999]:
                d[seqn]['calcium'] = np.nan
            d[seqn]['calcium'] /= 10

            # Creatinine:
            try:
                d[seqn]['creatinine'] = float(line[474:477])
            except ValueError:
                d[seqn]['creatinine'] = np.nan
            d[seqn]['creatinine_isUnacceptable'] = (d[seqn]['creatinine'] == 777)
            d[seqn]['creatinine_isTestnotdone'] = (d[seqn]['creatinine'] == 999)
            d[seqn]['creatinine_isBlank'] = np.isnan(d[seqn]['creatinine'])
            if d[seqn]['creatinine'] in [777, 999]:
                d[seqn]['creatinine'] = np.nan
            d[seqn]['creatinine'] /= 10

            # Serum potassium:
            try:
                d[seqn]['potassium'] = float(line[273:276])
            except ValueError:
                d[seqn]['potassium'] = np.nan
            d[seqn]['potassium_isUnacceptable'] = (d[seqn]['potassium'] == 888)
            d[seqn]['potassium_isBlank'] = np.isnan(d[seqn]['potassium'])
            if d[seqn]['potassium'] in [888]:
                d[seqn]['potassium'] = np.nan
            d[seqn]['potassium'] /= 10

            # Serum sodium:
            try:
                d[seqn]['sodium'] = float(line[270:273])
            except ValueError:
                d[seqn]['sodium'] = np.nan
            d[seqn]['sodium_isUnacceptable'] = (d[seqn]['sodium'] == 888)
            d[seqn]['sodium_isBlank'] = np.isnan(d[seqn]['sodium'])
            if d[seqn]['sodium'] in [888]:
                d[seqn]['sodium'] = np.nan

            # Total bilirubin:
            try: 
                d[seqn]['total_bilirubin'] = float(line[450:454])
            except ValueError:
                d[seqn]['total_bilirubin'] = np.nan
            d[seqn]['total_bilirubin_isUnacceptable'] = (d[seqn]['total_bilirubin'] == 7777)
            d[seqn]['total_bilirubin_isBlankbutapplicable'] = (d[seqn]['total_bilirubin'] == 8888)
            d[seqn]['total_bilirubin_isTestnotdone'] = (d[seqn]['total_bilirubin'] == 9999)
            d[seqn]['total_bilirubin_isBlank'] = np.isnan(d[seqn]['total_bilirubin'])
            if d[seqn]['total_bilirubin'] in [7777, 8888, 9999]:
                d[seqn]['total_bilirubin'] = np.nan
            d[seqn]['total_bilirubin'] /= 100

            # Serum protein
            try: 
                d[seqn]['serum_protein'] = float(line[226:230])
            except ValueError:
                d[seqn]['serum_protein'] = np.nan
            d[seqn]['serum_protein_isMissingAge1to3'] = (d[seqn]['serum_protein'] == 9999)
            d[seqn]['serum_protein_isBlank'] = np.isnan(d[seqn]['serum_protein'])
            if d[seqn]['serum_protein'] in [7777, 8888, 9999]:
                d[seqn]['serum_protein'] = np.nan
            d[seqn]['serum_protein'] /= 10
            try:
                isImputed = (int(line[230]) == 1)
            except ValueError:
                isImputed = False
            if isImputed:
                d[seqn]['serum_protein'] = np.nan
                # I assume that the study having imputed the value is the same 
                # as the study having left the field blank
                d[seqn]['serum_protein_isBlank'] = True

            # Red blood cell count
            try:
                d[seqn]['red_blood_cells'] = float(line[525:528])
            except ValueError:
                d[seqn]['red_blood_cells'] = np.nan
            d[seqn]['red_blood_cells_isUnacceptable'] = (d[seqn]['red_blood_cells'] == 777)
            d[seqn]['red_blood_cells_isBlankbutapplicable'] = (d[seqn]['red_blood_cells'] == 888)
            if d[seqn]['red_blood_cells'] in [777, 888]:
                d[seqn]['red_blood_cells'] = np.nan
            d[seqn]['red_blood_cells'] /= 100

            # White blood cell count
            try:
                d[seqn]['white_blood_cells'] = float(line[528:531])
            except ValueError:
                d[seqn]['white_blood_cells'] = np.nan
            d[seqn]['white_blood_cells_isUnacceptable'] = (d[seqn]['white_blood_cells'] == 777)
            d[seqn]['white_blood_cells_isBlankbutapplicable'] = (d[seqn]['white_blood_cells'] == 888)
            if d[seqn]['white_blood_cells'] in [777, 888]:
                d[seqn]['white_blood_cells'] = np.nan
            d[seqn]['white_blood_cells'] /= 10

            # Hemoglobin
            try: 
                d[seqn]['hemoglobin'] = float(line[246:250])
            except ValueError:
                d[seqn]['hemoglobin'] = np.nan
            d[seqn]['hemoglobin_isMissing'] = (d[seqn]['hemoglobin'] == 8888)
            d[seqn]['hemoglobin_isUnacceptable'] = (d[seqn]['hemoglobin'] == 7777)
            d[seqn]['hemoglobin_isBlank'] = np.isnan(d[seqn]['hemoglobin'])
            if d[seqn]['hemoglobin'] in [7777, 8888]:
                d[seqn]['hemoglobin'] = np.nan
            d[seqn]['hemoglobin'] /= 10
            try:
                isImputed = (int(line[250]) == 1)
            except ValueError:
                isImputed = False
            if isImputed:
                d[seqn]['hemoglobin'] = np.nan
                # I assume that the study having imputed the value is the same 
                # as the study having left the field blank
                d[seqn]['hemoglobin_isBlank'] = True

            # Hematocrit
            try: 
                d[seqn]['hematocrit'] = float(line[251:254])
            except ValueError:
                d[seqn]['hematocrit'] = np.nan
            d[seqn]['hematocrit_isUnacceptable'] = (d[seqn]['hematocrit'] == 777)
            d[seqn]['hematocrit_isMissing'] = (d[seqn]['hematocrit'] == 888)
            d[seqn]['hematocrit_isBlank'] = np.isnan(d[seqn]['hematocrit'])
            if d[seqn]['hematocrit'] in [777, 888]:
                d[seqn]['hematocrit'] = np.nan
            try:
                isImputed = (int(line[254]) == 1)
            except ValueError:
                isImputed = False
            if isImputed:
                d[seqn]['hematocrit'] = np.nan
                # I assume that the study having imputed the value is the same 
                # as the study having left the field blank
                d[seqn]['hematocrit_isBlank'] = True

            # Platelet estimate
            try:
                platelets = int(line[337]) 
            except:
                platelets = np.nan
            d[seqn]['platelets_isNormal'] = (platelets == 0)
            d[seqn]['platelets_isIncreased'] = (platelets == 2)
            d[seqn]['platelets_isDecreased'] = (platelets == 3)
            d[seqn]['platelets_isNoestimate'] = (platelets == 9) 
            d[seqn]['platelets_isBlank'] = np.isnan(platelets) 

            # Segmented neutrophils (mature)
            try:
                d[seqn]['segmented_neutrophils'] = float(line[320:322])
            except ValueError:
                d[seqn]['segmented_neutrophils'] = np.nan
            d[seqn]['segmented_neutrophils_isBlank'] = np.isnan(d[seqn]['segmented_neutrophils'])

            # Lymphocytes 
            try:
                d[seqn]['lymphocytes'] = float(line[326:328])
            except ValueError:
                d[seqn]['lymphocytes'] = np.nan
            d[seqn]['lymphocytes_isBlank'] = np.isnan(d[seqn]['lymphocytes'])

            # monocytes 
            try:
                d[seqn]['monocytes'] = float(line[328:330])
            except ValueError:
                d[seqn]['monocytes'] = np.nan
            d[seqn]['monocytes_isBlank'] = np.isnan(d[seqn]['monocytes'])

            # eosinophils 
            try:
                d[seqn]['eosinophils'] = float(line[322:324])
            except ValueError:
                d[seqn]['eosinophils'] = np.nan
            d[seqn]['eosinophils_isBlank'] = np.isnan(d[seqn]['eosinophils'])

            # basophils 
            try:
                d[seqn]['basophils'] = float(line[324:326])
            except ValueError:
                d[seqn]['basophils'] = np.nan
            d[seqn]['basophils_isBlank'] = np.isnan(d[seqn]['basophils'])

            # band_neutrophils 
            try:
                d[seqn]['band_neutrophils'] = float(line[318:320])
            except ValueError:
                d[seqn]['band_neutrophils'] = np.nan
            d[seqn]['band_neutrophils_isBlank'] = np.isnan(d[seqn]['band_neutrophils'])

            # Serum cholesterol
            try: 
                d[seqn]['cholesterol'] = float(line[236:240])
            except ValueError:
                d[seqn]['cholesterol'] = np.nan
            d[seqn]['cholesterol_isMissing'] = (d[seqn]['cholesterol'] == 8888)
            d[seqn]['cholesterol_isMissingAge1to3'] = (d[seqn]['cholesterol'] == 9999)
            d[seqn]['cholesterol_isBlank'] = np.isnan(d[seqn]['cholesterol'])
            if d[seqn]['cholesterol'] in [8888, 9999]:
                d[seqn]['cholesterol'] = np.nan
            try:
                isImputed = (int(line[240]) == 1)
            except ValueError:
                isImputed = False
            if isImputed:
                d[seqn]['cholesterol'] = np.nan
                # I assume that the study having imputed the value is the same 
                # as the study having left the field blank
                d[seqn]['cholesterol_isBlank'] = True

            # Urine albumin
            try: 
                urine_albumin = int(line[500])
            except ValueError:
                urine_albumin = np.nan
            d[seqn]['urine_albumin_isNegative'] = (urine_albumin == 0)
            d[seqn]['urine_albumin_is>=30'] = (urine_albumin == 1) 
            d[seqn]['urine_albumin_is>=100'] = (urine_albumin == 2)
            d[seqn]['urine_albumin_is>=300'] = (urine_albumin == 3) 
            d[seqn]['urine_albumin_is>=1000'] = (urine_albumin == 4) 
            d[seqn]['urine_albumin_isTrace'] = (urine_albumin == 5)
            d[seqn]['urine_albumin_isBlankbutapplicable'] = (urine_albumin == 8)
            d[seqn]['urine_albumin_isBlank'] = np.isnan(urine_albumin) 

            # Urine glucose
            try:
                urine_glucose = int(line[501])
            except ValueError:
                urine_glucose = np.nan
            d[seqn]['urine_glucose_isNegative'] = (urine_glucose == 0)
            d[seqn]['urine_glucose_isLight'] = (urine_glucose == 1)
            d[seqn]['urine_glucose_isMedium'] = (urine_glucose == 2)
            d[seqn]['urine_glucose_isDark'] = (urine_glucose == 3)
            d[seqn]['urine_glucose_isVerydark'] = (urine_glucose == 4)
            d[seqn]['urine_glucose_isTrace'] = (urine_glucose == 5)
            d[seqn]['urine_glucose_isBlankbutapplicable'] = (urine_glucose == 8)
            d[seqn]['urine_glucose_isBlank'] = np.isnan(urine_glucose)

            # Urine pH
            try:
                d[seqn]['urine_pH'] = int(line[502])
            except ValueError:
                d[seqn]['urine_pH'] = np.nan
            d[seqn]['urine_pH_isBlank'] = np.isnan(d[seqn]['urine_pH'])
            if d[seqn]['urine_pH'] == 4:
                d[seqn]['urine_pH_isBlankbutapplicable'] = True
                d[seqn]['urine_pH'] = np.nan
            else:
                d[seqn]['urine_pH_isBlankbutapplicable'] = False

            # Hematest
            try:
                urine_hematest = int(line[503])
            except ValueError:
                urine_hematest = np.nan
            d[seqn]['urine_hematest_isNegative'] = (urine_hematest == 0)
            d[seqn]['urine_hematest_isSmall'] = (urine_hematest == 1)
            d[seqn]['urine_hematest_isModerate'] = (urine_hematest == 2)
            d[seqn]['urine_hematest_isLarge'] = (urine_hematest == 3)
            d[seqn]['urine_hematest_isVerylarge'] = (urine_hematest == 4)
            d[seqn]['urine_hematest_isTrace'] = (urine_hematest == 5)
            d[seqn]['urine_hematest_isBlankbutapplicable'] = (urine_hematest == 8)
            d[seqn]['urine_hematest_isBlank'] = np.isnan(urine_hematest)

            # Sedimentation rate
            try:
                d[seqn]['sedimentation_rate'] = float(line[279:282])
            except ValueError:
                d[seqn]['sedimentation_rate'] = np.nan
            d[seqn]['sedimentation_rate_isBlank'] = np.isnan(d[seqn]['sedimentation_rate'])
            if d[seqn]['sedimentation_rate'] == 888:
                d[seqn]['sedimentation_rate_isBlankbutapplicable'] = True
                d[seqn]['sedimentation_rate'] = np.nan
            else:
                d[seqn]['sedimentation_rate_isBlankbutapplicable'] = False

            # Uric acid
            try: 
                d[seqn]['uric_acid'] = float(line[462:465])
            except ValueError:
                d[seqn]['uric_acid'] = np.nan
            d[seqn]['uric_acid_isUnacceptable'] = (d[seqn]['uric_acid'] == 777)
            d[seqn]['uric_acid_isBlankbutapplicable'] = (d[seqn]['uric_acid'] == 888)
            d[seqn]['uric_acid_isTestnotdone'] = (d[seqn]['uric_acid'] == 999)
            d[seqn]['uric_acid_isBlank'] = np.isnan(d[seqn]['uric_acid'])
            if d[seqn]['uric_acid'] in [777, 888, 999]:
                d[seqn]['uric_acid'] = np.nan
            d[seqn]['uric_acid'] /= 10

    with open(medexamtapepath, 'r') as handle:
        for line in handle:
            seqn = int(line[:5])

            # Systolic blood pressure
            try:
                d[seqn]['systolic_blood_pressure'] = int(line[227:230])
            except ValueError:
                d[seqn]['systolic_blood_pressure'] = np.nan

            d[seqn]['systolic_blood_pressure_isBlank'] = (d[seqn]['systolic_blood_pressure'] == 888)
            d[seqn]['systolic_blood_pressure_isAgeUnder6'] = (d[seqn]['systolic_blood_pressure'] == 999)
            if d[seqn]['systolic_blood_pressure'] in [888, 999]:
                d[seqn]['systolic_blood_pressure'] = np.nan

            # Pulse pressure
            try:
                diastolic = int(line[230:233])
            except ValueError:
                diastolic = np.nan
            # For this case, we have to treat "blank but applicable" and 
            # "age under 6" the same as blank
            if diastolic in [888, 999]:
                diastolic = np.nan
            if np.isnan(d[seqn]['systolic_blood_pressure']) or np.isnan(diastolic):
                d[seqn]['pulse_pressure'] = np.nan
            else:
                d[seqn]['pulse_pressure'] = d[seqn]['systolic_blood_pressure'] - diastolic
            
            ## Obesity
            #try:
            #    obesity = int(line[360])
            #except ValueError:
            #    obesity = np.nan

            #if obesity == 1:
            #    d[seqn]['obesity'] = 1.0
            #elif obesity == 2:
            #    d[seqn]['obesity'] = 0.0
            #else:
            #    d[seqn]['obesity'] = np.nan

            #d[seqn]['obesity_isBlankButApplicable'] = (obesity == 8)
            #d[seqn]['obesity_isBlank'] = np.isnan(obesity)

    with open(anthropometrypath, 'r') as handle:
        for line in handle:
            seqn = int(line[0:5])

            # weight
            try:
                d[seqn]['weight'] = int(line[259:264])
            except ValueError:
                d[seqn]['weight'] = np.nan 
            if d[seqn]['weight'] == 88888:
                d[seqn]['weight'] = np.nan
            # Here we group together the 98 participants with imputed weights
            # and the 4 participants for whom the field is "blank but 
            # applicable"
            weightIsImputed = (int(line[264]) == 1)
            if weightIsImputed:
                d[seqn]['weight'] = np.nan
            d[seqn]['weight_isBlank'] = np.isnan(d[seqn]['weight'])
            d[seqn]['weight'] /= 100

            # height
            try:
                d[seqn]['height'] = int(line[265:269])
            except ValueError:
                d[seqn]['height'] = np.nan
            if d[seqn]['height'] == 8888:
                d[seqn]['height'] = np.nan

            # Here we group together the 60 participants with imputed heights
            # and the 4 participants for whom the field is "blank but 
            # applicable"
            heightIsImputed = (int(line[272]) == 1)
            if heightIsImputed:
                d[seqn]['height'] = np.nan
            d[seqn]['height_isBlank'] = np.isnan(d[seqn]['height'])
            d[seqn]['height'] /= 10

    d2 = {}
    with open(vitlpath, 'r') as handle:
        for line in handle:
            seqn = int(line[11:16])
            try: 
                d[seqn]
            except KeyError:
                continue
            d2[seqn] = {}
            d2[seqn]['month_last_known_alive'] = int(line[17:19])
            d2[seqn]['month_last_known_alive_isDontknow'] = (d2[seqn]['month_last_known_alive'] == 98)
            d2[seqn]['month_last_known_alive_isNotascertained'] = (d2[seqn]['month_last_known_alive'] == 99)
            if d2[seqn]['month_last_known_alive'] in [98,99]:
                d2[seqn]['month_last_known_alive'] = np.nan

            d2[seqn]['year_last_known_alive'] = int(line[21:23])

            try:
                d2[seqn]['month_deceased'] = int(line[60:62])
            except ValueError:
                d2[seqn]['month_deceased'] = np.nan
            try:
                d2[seqn]['year_deceased'] = int(line[64:66])
            except ValueError:
                d2[seqn]['year_deceased'] = np.nan 
            d2[seqn]['month_deceased_isDontknow'] = (d2[seqn]['month_deceased'] == 98)
            d2[seqn]['month_deceased_isNotascertained'] = (d2[seqn]['month_deceased'] == 99)
            d2[seqn]['month_deceased_isBlank'] = np.isnan(d2[seqn]['month_deceased'])
            if d2[seqn]['month_deceased'] in [98, 99]:
                d2[seqn]['month_deceased'] = np.nan
            d2[seqn]['year_deceased_isBlank'] = np.isnan(d2[seqn]['year_deceased'])

    for seqn in list(d.keys()):
        try:
            d2[seqn]
        except KeyError:
            d[seqn]['survived_15_years'] = np.nan
            del d[seqn]['exam_year']
            del d[seqn]['exam_month']
            continue

        alive_time = d2[seqn]['year_last_known_alive']+d2[seqn]['month_last_known_alive']/12.0
        dead_time = d2[seqn]['year_deceased']+d2[seqn]['month_deceased']/12.0
        exam_time = d[seqn]['exam_year']+d[seqn]['exam_month']/12.0
        
        if np.isnan(dead_time):
            if alive_time > exam_time:
                years = -(alive_time - exam_time)
        else:
            if dead_time >= alive_time and dead_time > exam_time:
                years = dead_time - exam_time

        d[seqn]['survived_15_years'] = years
        # maxalive = max(
        #     d2[seqn]['year_last_known_alive']+d2[seqn]['month_last_known_alive']/12.0,
        #     d2[seqn]['year_deceased']+d2[seqn]['month_deceased']/12.0
        # )
        # years = maxalive- d[seqn]['exam_year']+d[seqn]['exam_month']/12.0
        #print (d[seqn]['exam_month'], d2[seqn]['month_last_known_alive'], d2[seqn]['month_deceased'])
        

        # if d2[seqn]['year_last_known_alive'] - d[seqn]['exam_year'] > 15:
        #     d[seqn]['survived_15_years'] = True

        # elif d2[seqn]['year_last_known_alive'] - d[seqn]['exam_year'] == 15\
        #         and d2[seqn]['month_last_known_alive'] >= d[seqn]['exam_month']:
        #     d[seqn]['survived_15_years'] = True

        # elif d2[seqn]['year_deceased'] - d[seqn]['exam_year'] < 15:
        #     d[seqn]['survived_15_years'] = False

        # elif d2[seqn]['year_deceased'] - d[seqn]['exam_year'] == 15\
        #         and d2[seqn]['month_deceased'] < d[seqn]['exam_month']:
        #     d[seqn]['survived_15_years'] = False
        # else:
        #     d[seqn]['survived_15_years'] = np.nan
                
        del d[seqn]['exam_year']
        del d[seqn]['exam_month']

    for subd in d.values():
        if type(subd) != dict:
            print(subd)
    dataframe = pd.DataFrame.from_dict(d, orient='index')
    y = np.array(dataframe['survived_15_years'], dtype=float)
    del dataframe['survived_15_years']

    # Remove participants with 'NaN' labels
    bad_idxs = list(np.where(np.isnan(y))[0])
    dataframe.drop(dataframe.index[bad_idxs], inplace=True)
    y = y[~np.isnan(y)]
    
    for colname in list(dataframe.columns):
        if np.all(dataframe[colname] == dataframe[colname].iloc[0]):
            dataframe.drop(colname, inplace=True, axis=1)
    for colname in list(dataframe.columns):
        assert not np.all(dataframe[colname] == dataframe[colname].iloc[0])
    
    # clean up a bit
    for c in dataframe.columns:
        if c.endswith("_isBlank"):
            del dataframe[c]   
    dataframe["bmi"] = 10000 * dataframe["weight"].values.copy() / (dataframe["height"].values.copy() * dataframe["height"].values.copy())
    del dataframe["weight"]
    del dataframe["height"]
    del dataframe["urine_hematest_isTrace"] # would have no variance in the train set
    del dataframe["SGOT_isBlankbutapplicable"] # would have no variance in the train set
    del dataframe["calcium_isBlankbutapplicable"] # would have no variance in the train set
    del dataframe["uric_acid_isBlankbutapplicable"] # would only have one true value in the train set
    del dataframe["urine_hematest_isVerylarge"] # would only have one true value in the train set
    del dataframe["total_bilirubin_isBlankbutapplicable"] # would only have one true value in the train set
    del dataframe["alkaline_phosphatase_isBlankbutapplicable"] # would only have one true value in the train set
    del dataframe["hemoglobin_isUnacceptable"] # redundant with hematocrit_isUnacceptable

    # remove not interpretable features
    del dataframe["alkaline_phosphatase_isUnacceptable"]
    del dataframe["alkaline_phosphatase_isTestnotdone"]
    del dataframe["SGOT_isUnacceptable"]
    del dataframe["SGOT_isTestnotdone"]
    del dataframe["BUN_isUnacceptable"]
    del dataframe["BUN_isTestnotdone"]
    del dataframe["calcium_isUnacceptable"]
    del dataframe["calcium_isTestnotdone"]
    del dataframe["creatinine_isUnacceptable"]
    del dataframe["creatinine_isTestnotdone"]
    del dataframe["potassium_isUnacceptable"]
    del dataframe["sodium_isUnacceptable"]
    del dataframe["total_bilirubin_isUnacceptable"]
    del dataframe["total_bilirubin_isTestnotdone"]
    del dataframe["red_blood_cells_isUnacceptable"]
    del dataframe["red_blood_cells_isBlankbutapplicable"]
    del dataframe["white_blood_cells_isUnacceptable"]
    del dataframe["white_blood_cells_isBlankbutapplicable"]
    del dataframe["hemoglobin_isMissing"]
    del dataframe["hematocrit_isUnacceptable"]
    del dataframe["hematocrit_isMissing"]
    del dataframe["cholesterol_isMissing"]
    del dataframe["urine_albumin_isBlankbutapplicable"]
    del dataframe["urine_glucose_isBlankbutapplicable"]
    del dataframe["urine_pH_isBlankbutapplicable"]
    del dataframe["urine_hematest_isBlankbutapplicable"]
    del dataframe["sedimentation_rate_isBlankbutapplicable"]
    del dataframe["uric_acid_isUnacceptable"]
    del dataframe["uric_acid_isTestnotdone"]


    rows = np.where(np.invert(np.isnan(dataframe["systolic_blood_pressure"]) | np.isnan(dataframe["bmi"])))[0]
    dataframe = dataframe.iloc[rows,:]
    y = y[rows]

    name_map = {
        "sex_isFemale": "Sex",
        "age": "Age",
        "systolic_blood_pressure": "Systolic blood pressure",
        "bmi": "BMI",
        "white_blood_cells": "White blood cells", # (mg/dL)
        "sedimentation_rate": "Sedimentation rate",
        "serum_albumin": "Blood albumin",
        "alkaline_phosphatase": "Alkaline phosphatase",
        "cholesterol": "Total cholesterol",
        "physical_activity": "Physical activity",
        "hematocrit": "Hematocrit",
        "uric_acid": "Uric acid",
        "red_blood_cells": "Red blood cells",
        "urine_albumin_isNegative": "Albumin present in urine",
        "serum_protein": "Blood protein"
    }
    feat_names = list(map(lambda x: name_map.get(x, x), dataframe.columns))
    X = SimpleImputer().fit_transform(dataframe)
    y = y.reshape(-1, 1)
    rand_idx = np.arange(X.shape[0])
    np.random.shuffle(rand_idx)
    X = X[rand_idx, :]
    y = y[rand_idx, :]

    # Determine whether a feature is categorical
    categorical = np.zeros(X.shape[1], dtype=bool)
    for i in range(X.shape[1]):
        if is_discrete_array(X[:, i]) and np.unique(X[:, i]).size == 2:
            categorical[i] = True

    # Rescale continuous features
    continuous_indices = [i for i in range(X.shape[1]) if not categorical[i]]
    if len(continuous_indices) > 0:
        X[:, continuous_indices] = MaxAbsScaler().fit_transform(X[:, continuous_indices])

    # print("number of people surviving ", (y < 0).sum())
    # print("number of people not surviving ", (y > 0).sum())

    return X, y, feat_names

def get_mortality_gt():
    return ["potassium", "urine_pH", "Age", "creatinine", "BUN", "sodium"], [
            {"potassium", "urine_pH"}, {"Age", "potassium"}, 
            {"creatinine", "potassium"}, {"BUN", "creatinine"}, 
            {"BUN", "Age"}, {"BUN", "potassium"}, {"BUN", "sodium"},
            {"Age", "urine_pH"}, {"creatinine", "sodium"}, {"potassium", "sodium"}
        ]
        
def load_diabetes_data():
    diabetes = load_diabetes()
    X = diabetes.data
    Y = diabetes.target.reshape(-1, 1)
    feat_names = diabetes.feature_names

    return X, Y, feat_names

def load_cal_housing_data():
    cal_housing = fetch_california_housing()
    X = cal_housing.data
    Y = cal_housing.target.reshape(-1, 1)
    feat_names = cal_housing.feature_names

    return X, Y, feat_names
