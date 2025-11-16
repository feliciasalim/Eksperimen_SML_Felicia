import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

def preprocessing_data(df):
    df = df.copy()
    
    # Drop id 
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Mapping 
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1, 'Other': 2})
    df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
    df['Residence_type'] = df['Residence_type'].map({'Rural': 0, 'Urban': 1})
    df['work_type'] = df['work_type'].map({
        'Govt_job': 0,
        'Never_worked': 1, 
        'Private': 2,
        'Self-employed': 3,
        'children': 4
    })
    df['smoking_status'] = df['smoking_status'].map({
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': 3
    })
    
    #Drop missing & duplicates
    df = df.dropna()
    df = df.drop_duplicates()
    
    #Drop outlier
    outlier_cols = ['bmi', 'avg_glucose_level']
    Q1 = df[outlier_cols].quantile(0.25)
    Q3 = df[outlier_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    for col in outlier_cols:
        df = df[(df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])]
    
    #Pisah target class
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    #Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
    os.makedirs("stroke_preprocessing", exist_ok=True)
    
    return df_train, df_test


if __name__ == "__main__":
    df = pd.read_csv("../stroke_raw.csv") 
    df_train, df_test = preprocessing_data(df)
    df_train.to_csv("stroke_preprocessing/data_train.csv", index=False)
    df_test.to_csv("stroke_preprocessing/data_test.csv", index=False)
    