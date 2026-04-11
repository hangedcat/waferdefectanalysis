import numpy as np
import pandas as pd
from pandas import DataFrame
from collections import defaultdict
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score


def generate_wafer_data() -> DataFrame:
    np.random.seed(42)
    wafer_id = [f"W{i:03d}"for i in range(1, 501)]
    lot_id = [f'L{i//50 + 1:03d}' for i in range(0, 500)]
    defect_count = np.random.randint(0, 20, size=500)
    defect_density = defect_count/700
    temperature = np.random.normal(225, 25, 500)
    pressure = np.random.normal(2, 0.3, 500)
    passed = defect_count < 10

    df_dict = {'wafer_id':wafer_id,
            'lot_id':lot_id,
            'defect_count':defect_count,
            "defect_density":defect_density,
            'temperature':temperature,
            'pressure':pressure,
            'passed':passed
            }

    return pd.DataFrame(df_dict)

def optimize_memory(data: DataFrame) -> DataFrame:
    data = data.copy()
    print(f'Memory usage before: {data.memory_usage(deep=True).sum() / 1024}KB')
    registered = defaultdict(list)
    for column in data.columns:
        registered[data[column].dtype].append(column)
    for key, columns in registered.items():
        for column in columns:
            if pd.api.types.is_object_dtype(key) or pd.api.types.is_string_dtype(key):
                if (data.shape[0]//len(data[column].unique())) > 5 :
                    data[column] = data[column].astype('category')
            elif pd.api.types.is_integer_dtype(key):
                if len(data[column].unique()) < 127:
                    data[column] = pd.to_numeric(data[column], downcast='integer')
    print(f'Memory usage after: {data.memory_usage(deep=True).sum() / 1024}KB')
    return data

def analyze_lots(data: DataFrame, column: str= "defect_count", groupby: str = 'lot_id') -> tuple[DataFrame, DataFrame]:
    data = data.copy()
    agg_data = data.groupby(groupby).agg(
                                        mean_defect = (column, 'mean'),
                                        max_defects=(column, 'max'),
                                        min_defects=(column, 'min'),
                                        std_defects=(column, 'std'),
                                        pass_rate=('passed', 'mean')
                                        )
    data['lot_avg_defect'] = data.groupby(groupby)[column].transform('mean')
    return agg_data, data

def engineer_features(data: DataFrame):
    data = data.copy()
    data['defect_above_lot_avg'] = data['defect_count'] > data['lot_avg_defect']
    data['temp_pressure_ratio'] = data['temperature']/data['pressure']
    data['defect_zscore'] = data.groupby('lot_id')['defect_count'].transform(lambda x: (x - x.mean()) / x.std())
    return data

def build_pipeline(data: DataFrame, target: str, features: list[str]) -> tuple[Pipeline, DataFrame, DataFrame, DataFrame, DataFrame, np.ndarray]:
    data = data.copy()
    y = data[target]
    X = data[features] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    my_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced'))
    ])
    my_pipe.fit(X_train, y_train)
    scores = cross_val_score(my_pipe, X, y, cv=5, scoring='accuracy')
    return my_pipe, X_train, y_train, X_test, y_test, scores

def evaluate_pipeline(data: DataFrame, target: str, features: list[str]):
    pipeline, X_train, y_train, X_test, y_test, scores = build_pipeline(data = data, target = target, features = features)
    y_pred = pipeline.predict(X = X_test)
    minority_recall = recall_score(y_true= y_test, y_pred= y_pred, pos_label=False)
    print(f'Accuracy: {accuracy_score(y_true = y_test, y_pred = y_pred)}\n')
    print(f'Classification Report: {classification_report(y_true = y_test, y_pred = y_pred)}\n')
    print(f'Confusion Matrix: {confusion_matrix(y_true= y_test, y_pred= y_pred)}')
    print(f'CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}')
    print(f'Defect detection rate (minority recall): {minority_recall:.4f}')

    return pipeline

def clean_data(data: DataFrame, config: dict[str, str]):
    data = data.copy()
    columns = config.keys()
    for i in columns:
        if config[i].lower() == 'drop':
            data = data.dropna(subset=[i])
        else:
            data[i] = data[i].fillna(data[i].agg(config[i]))
    
    return data

if __name__ == '__main__':
    df = generate_wafer_data()
    df.to_csv('df.csv', index=False)