import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def profile_data(df):
    return {
        'rows': int(len(df)), 'columns': int(len(df.columns)),
        'numeric_columns': [str(c) for c in df.select_dtypes(include='number').columns],
        'categorical_columns': [str(c) for c in df.select_dtypes(exclude='number').columns],
        'missing_values': {str(c): int(v) for c, v in df.isna().sum().items()},
        'constant_columns': [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1],
        'duplicate_rows': int(df.duplicated().sum()),
        'duplicate_columns': [str(df.columns[i]) for i in range(len(df.columns)) if any(df.iloc[:, i].equals(df.iloc[:, j]) for j in range(i))],
    }


def prepare_frames(train, test):
    train = train.copy()
    test = test.copy()
    # Preserve alignment even if evaluation data has unseen or missing columns.
    test = test.reindex(columns=train.columns)
    drop = [c for c in train if train[c].nunique(dropna=False) <= 1]
    drop += [c for i, c in enumerate(train.columns) if any(train[c].equals(train.iloc[:, j]) for j in range(i))]
    # IDs generally generalize poorly; avoid high-cardinality near-unique strings.
    drop += [c for c in train.select_dtypes(exclude='number') if len(train) > 30 and train[c].nunique(dropna=True) / max(len(train), 1) > .98]
    drop = list(dict.fromkeys(drop))
    train, test = train.drop(columns=drop), test.drop(columns=drop)
    if train.shape[1] == 0:
        train['__constant__'] = 0
        test['__constant__'] = 0
    for c in train.columns:
        if not pd.api.types.is_numeric_dtype(train[c]):
            train[c] = train[c].fillna('__MISSING__').astype(str)
            test[c] = test[c].fillna('__MISSING__').astype(str)
        else:
            train[c] = pd.to_numeric(train[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
            test[c] = pd.to_numeric(test[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return train, test


def make_preprocessor(x, scale=False):
    numeric = list(x.select_dtypes(include='number').columns)
    categorical = [c for c in x if c not in numeric]
    transformers = []
    if numeric:
        steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
        if scale:
            steps.append(('scaler', StandardScaler()))
        transformers.append(('num', Pipeline(steps), numeric))
    if categorical:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=2, max_categories=40, sparse_output=False))
        ]), categorical))
    return ColumnTransformer(transformers, sparse_threshold=0)


def supervised(train, test, target, task, budget):
    if target not in train:
        raise ValueError(f'Target column {target!r} not present in training CSV')
    labeled = train[target].notna()
    y = train.loc[labeled, target]
    if len(y) < 2:
        raise ValueError('At least two labeled rows are required')
    x_train, x_test = prepare_frames(train.loc[labeled].drop(columns=[target]), test.drop(columns=[target], errors='ignore'))
    is_cls = task == 'classification'
    if is_cls and y.nunique() < 2:
        model = DummyClassifier(strategy='most_frequent')
        pipe = Pipeline([('prep', make_preprocessor(x_train)), ('model', model)])
        pipe.fit(x_train, y)
        return pipe.predict(x_test), {'model': 'DummyClassifier', 'validation_score': None}
    if not is_cls and y.nunique() < 2:
        model = DummyRegressor(strategy='mean')
        pipe = Pipeline([('prep', make_preprocessor(x_train)), ('model', model)])
        pipe.fit(x_train, y)
        return pipe.predict(x_test), {'model': 'DummyRegressor', 'validation_score': None}
    stratify = y if is_cls and y.value_counts().min() >= 2 and y.nunique() <= max(2, int(len(y) * .15)) else None
    try:
        a, b = train_test_split(np.arange(len(y)), test_size=.2, random_state=RANDOM_STATE, stratify=stratify)
    except ValueError:
        a, b = train_test_split(np.arange(len(y)), test_size=.2, random_state=RANDOM_STATE)
    candidates = ([
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=90, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
        ('RandomForest', RandomForestClassifier(n_estimators=90, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
    ] if is_cls else [
        ('ExtraTrees', ExtraTreesRegressor(n_estimators=90, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
        ('RandomForest', RandomForestRegressor(n_estimators=90, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    best_name, best_score = None, -np.inf
    start = time.monotonic()
    for name, estimator in candidates:
        if time.monotonic() - start > budget * .6 and best_name is not None:
            break
        pipe = Pipeline([('prep', make_preprocessor(x_train)), ('model', estimator)])
        try:
            pipe.fit(x_train.iloc[a], y.iloc[a])
            prediction = pipe.predict(x_train.iloc[b])
            score = balanced_accuracy_score(y.iloc[b], prediction) if is_cls else -float(np.sqrt(mean_squared_error(y.iloc[b], prediction)))
            if np.isfinite(score) and score > best_score:
                best_name, best_score = name, score
        except (ValueError, MemoryError):
            continue
    if best_name is None:
        raise RuntimeError('All candidate models failed; check input columns and data types')
    estimator = dict(candidates)[best_name]
    final = Pipeline([('prep', make_preprocessor(x_train)), ('model', estimator)])
    final.fit(x_train, y)
    return final.predict(x_test), {'model': best_name, 'validation_score': float(best_score), 'metric': 'balanced_accuracy' if is_cls else 'negative_RMSE'}


def cluster(train, test, n_clusters):
    if n_clusters is None or n_clusters < 1:
        raise ValueError('Clustering requires --clusters N')
    x_train, x_test = prepare_frames(train, test)
    pipe = make_preprocessor(x_train, scale=True)
    a = pipe.fit_transform(x_train)
    b = pipe.transform(x_test)
    # Cap extreme magnitudes from unusual numeric inputs.
    a = np.clip(a, -1e5, 1e5)
    b = np.clip(b, -1e5, 1e5)
    if len(a) < n_clusters:
        raise ValueError('Number of training rows must be >= number of clusters')
    model = MiniBatchKMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=3, batch_size=256) if len(a) > 10000 else KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=5)
    model.fit(a)
    return model.predict(b), {'model': type(model).__name__, 'n_clusters': n_clusters}


def run(args):
    started = time.monotonic()
    train = pd.read_csv(args.train, low_memory=False)
    test = pd.read_csv(args.test, low_memory=False) if args.test else train.copy()
    profile = profile_data(train)
    if args.task == 'clustering':
        predictions, details = cluster(train, test, args.clusters)
    else:
        if not args.target:
            raise ValueError('Classification and regression require --target COLUMN')
        predictions, details = supervised(train, test, args.target, args.task, args.budget)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'prediction': predictions}).to_csv(out, index=False)
    result = {'task': args.task, 'data_profile': profile, 'model': details, 'prediction_count': int(len(predictions)), 'elapsed_seconds': round(time.monotonic()-started, 3)}
    profile_path = Path(args.profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'predictions': str(out), 'profile': str(profile_path), 'model': details, 'elapsed_seconds': result['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EECE 490 Unknown Dataset Hackathon provisional CLI')
    parser.add_argument('--train', required=True, help='Training CSV path')
    parser.add_argument('--test', help='Evaluation CSV path; if absent, predicts training rows')
    parser.add_argument('--task', required=True, choices=['classification', 'regression', 'clustering'])
    parser.add_argument('--target', help='Label column for supervised tasks')
    parser.add_argument('--clusters', type=int, help='Required number of clusters')
    parser.add_argument('--output', default='predictions.csv')
    parser.add_argument('--profile', default='profile.json')
    parser.add_argument('--budget', type=float, default=40.0)
    run(parser.parse_args())
