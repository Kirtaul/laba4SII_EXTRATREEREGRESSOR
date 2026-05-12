import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
#Загрузка данных
print("train.csv")
train = pd.read_csv('train.csv', sep='\t', header=None)
X_num = train.iloc[:, :-1].values
y_num = train.iloc[:, -1].values
print(f"train: {X_num.shape[0]} строк, {X_num.shape[1]} признаков")

print("\ntest.csv")
test = pd.read_csv('test.csv', sep='\t', header=None)
X_test_num = test.values
print(f"test: {X_test_num.shape[0]} строк, {X_test_num.shape[1]} признаков")

print("\nmoscow_dataset_2020.csv")
flats = pd.read_csv('moscow_dataset_2020.csv')
X_flat = flats.drop('price', axis=1)
y_flat = flats['price'].values
print(f"Квартиры: {X_flat.shape[0]} строк, {X_flat.shape[1]} столбцов")
#Предобработка числового датасета
def preprocess_numeric(X, y, X_test=None):
    Q1, Q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    mask = np.all((X >= Q1 - 1.5*(Q3-Q1)) & (X <= Q3 + 1.5*(Q3-Q1)), axis=1)
    X_clean, y_clean = X[mask], y[mask]
    print(f"Удалено выбросов: {X.shape[0] - X_clean.shape[0]} (осталось {X_clean.shape[0]})")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_scaled, y_clean, X_test_scaled
    return X_scaled, y_clean

#Предобработка квартирного датасета
def preprocess_categorical(df):
    X = df.drop('price', axis=1)
    y = df['price'].values
    num_cols = ['floorNumber', 'floorsTotal', 'totalArea', 'kitchenArea', 'latitude', 'longitude']
    X_num = X[num_cols].values
    Q1, Q3 = np.percentile(X_num, 25, axis=0), np.percentile(X_num, 75, axis=0)
    mask = np.all((X_num >= Q1 - 1.5*(Q3-Q1)) & (X_num <= Q3 + 1.5*(Q3-Q1)), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    print(f"Удалено выбросов: {X.shape[0] - X_clean.shape[0]}")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['wallsMaterial'])
    ])
    X_trans = preprocessor.fit_transform(X_clean)
    return X_trans, y_clean

#Обучение регрессии и оценка
def train_regression(X, y, dataset_name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    etr = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(etr, param_grid, cv=3, scoring='r2', verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print(f"{dataset_name} – лучшие параметры: {grid.best_params_}")

    y_pred = best.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(best, X, y, cv=kfold, scoring='r2')
    print(f"MSE={mse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
    print(f"CV R2: mean={cv.mean():.4f}, std={cv.std():.4f}")

    # график
    plt.figure(figsize=(5,4))
    plt.scatter(y_val, y_pred, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel("Реальная цена")
    plt.ylabel("Предсказанная цена")
    plt.title(dataset_name)
    plt.tight_layout()
    plt.show()

    # обучение на всех данных для предсказаний на тесте
    best_full = ExtraTreesRegressor(**grid.best_params_, random_state=42, n_jobs=-1)
    best_full.fit(X, y)
    return best_full

#Кластеризация с автоматическим выбором k
def cluster_data(X, title, y_price=None):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sil_scores = []
    best_k, best_sil = 2, -1
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        sil_scores.append(sil)
        if sil > best_sil:
            best_sil, best_k = sil, k
    print(f"Лучшее k = {best_k}, силуэт = {best_sil:.4f}")

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    if y_price is not None:
        print("Средняя цена по кластерам:")
        for i in range(best_k):
            mask = clusters == i
            print(f"Кластер {i}: {mask.sum()} объектов, цена = {np.mean(y_price[mask]):.2f}")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(5,4))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap='viridis', alpha=0.7)
    plt.title(f"{title}, k={best_k}")
    plt.tight_layout()
    plt.show()

#Основной блок
print("\nЧисловой датасет")
X_num_clean, y_num_clean, X_test_clean = preprocess_numeric(X_num, y_num, X_test_num)
model_num = train_regression(X_num_clean, y_num_clean, "Числовой датасет")
y_test_pred_num = model_num.predict(X_test_clean)
print("\nПредсказания для test.csv:")
print(f"  Среднее = {np.mean(y_test_pred_num):.4f}, std = {np.std(y_test_pred_num):.4f}")
print(f"  Диапазон = [{np.min(y_test_pred_num):.4f}, {np.max(y_test_pred_num):.4f}]")
print("  Первые 10 предсказаний:", y_test_pred_num[:10])

cluster_data(X_num_clean, "Числовой датасет", y_price=y_num_clean)

#Квартирный датасет
print("\nКвартирный датасет")
X_flat_clean, y_flat_clean = preprocess_categorical(flats)
model_flat = train_regression(X_flat_clean, y_flat_clean, "Квартиры Москвы")
cluster_data(X_flat_clean, "Квартиры Москвы", y_price=y_flat_clean)
