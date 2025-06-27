import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Dosya yolunu güncelle
dosya_yolu = r"C:\Users\Administrator\Desktop\VerilerDüzenlenmişVeriseti.csv"

# CSV dosyasını oku
df = pd.read_csv(dosya_yolu)

# İlk 5 satırı görüntüle
print(df.head())

# Veri setinin genel bilgilerini görüntüle
print("\nVeri seti bilgileri:")
print(df.info())

# Eksik değerleri kontrol et
print("\nEksik değerler:")
print(df.isnull().sum())


# "Oda Sayısı" sütununu sayıya çevirme fonksiyonu
def oda_sayisini_donustur(oda):
    if "+" in oda:
        odalar = oda.split("+")
        try:
            return float(odalar[0]) + float(odalar[1])  # "2.5+1" gibi durumları da işler
        except ValueError:
            return None  # Hatalı veri varsa None (eksik) olarak işaretle
    elif oda.replace(".", "").isdigit():  
        return float(oda)  # Eğer direkt sayıysa (2.5 gibi)
    else:
        return None  # Anlamlandıramadığımız verileri boş olarak bırak

# Dönüştürme işlemi
df["Oda Sayısı"] = df["Oda Sayısı"].apply(oda_sayisini_donustur)

# Eksik kalan değerleri kontrol et
print("\nEksik değerler (dönüştürmeden sonra):")
print(df["Oda Sayısı"].isnull().sum())

# Güncellenmiş veri tiplerini kontrol et
print("\nGüncellenmiş veri türleri:")
print(df.dtypes)

# İlk birkaç satırı tekrar görüntüle
print("\nDönüştürülmüş veri:")
print(df.head())

print("\nEksik değerler (Oda Sayısı sütunu):")
print(df["Oda Sayısı"].isnull().sum())  # Eksik değerleri gösterir

# Eksik değerleri en yaygın değer (mode) ile dolduralım
df["Oda Sayısı"] = df["Oda Sayısı"].fillna(df["Oda Sayısı"].mode()[0])

# Kontrol edelim
print("\nEksik değerler (düzeltildikten sonra):")
print(df["Oda Sayısı"].isnull().sum())  # 0 çıkmalı!



plt.figure(figsize=(10, 5))
sns.histplot(df["Fiyat (TL)"], bins=50, kde=True)
plt.xlabel("Fiyat (TL)")
plt.ylabel("İlan Sayısı")
plt.title("Fiyat Dağılımı")
plt.show()


plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["Büyüklük (m²)"], y=df["Fiyat (TL)"])
plt.xlabel("Büyüklük (m²)")
plt.ylabel("Fiyat (TL)")
plt.title("Büyüklük ve Fiyat İlişkisi")
plt.show()


plt.figure(figsize=(10, 5))
sns.boxplot(x=df["Oda Sayısı"], y=df["Fiyat (TL)"])
plt.xlabel("Oda Sayısı")
plt.ylabel("Fiyat (TL)")
plt.title("Oda Sayısı ve Fiyat İlişkisi")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
ortalama_fiyatlar = df.groupby("Şehir")["Fiyat (TL)"].mean().sort_values(ascending=False)
sns.barplot(x=ortalama_fiyatlar.index, y=ortalama_fiyatlar.values)
plt.xlabel("Şehir")
plt.ylabel("Ortalama Fiyat (TL)")
plt.title("Şehirlere Göre Ortalama Ev Fiyatları")
plt.xticks(rotation=90)
plt.show()


# Fiyat sütununu sıralayarak en yüksek değerlere bakalım
print(df["Fiyat (TL)"].describe())

# Üst limit belirleyelim (örneğin 99. persentil)
upper_limit = df["Fiyat (TL)"].quantile(0.99)

# Aşırı yüksek fiyatları filtreleyelim
df = df[df["Fiyat (TL)"] <= upper_limit]

# Tekrar fiyat dağılımını çizelim


plt.figure(figsize=(10,5))
sns.histplot(df["Fiyat (TL)"], bins=50, kde=True)
plt.xlabel("Fiyat (TL)")
plt.ylabel("İlan Sayısı")
plt.title("Temizlenmiş Fiyat Dağılımı")
plt.show()


# Fiyat sütununa log dönüşümü uygulayalım
df["Log_Fiyat"] = np.log1p(df["Fiyat (TL)"])

# Logaritmik dönüşüm sonrası fiyat dağılımını çizelim
plt.figure(figsize=(10,5))
sns.histplot(df["Log_Fiyat"], bins=50, kde=True)
plt.xlabel("Log(Fiyat)")
plt.ylabel("İlan Sayısı")
plt.title("Logaritmik Dönüşüm Uygulanmış Fiyat Dağılımı")
plt.show()





# Bağımlı değişken (hedef)
y = df["Log_Fiyat"]  # Log dönüşümü yapılmış fiyatı kullanıyoruz

# Bağımsız değişkenler (özellikler)
X = df[["Büyüklük (m²)", "Oda Sayısı"]]  # Şehir gibi kategorik değişkenleri sonra ekleyebiliriz

# Eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")




# Bağımsız değişkenler (X) ve bağımlı değişken (y) seçimi
X = df[['Büyüklük (m²)', 'Oda Sayısı', 'Bina Yaşı (yıl)']]  # Yeni değişkenler eklendi!
y = df["Log_Fiyat"]  # Log dönüşümlü fiyatı tahmin ediyoruz

# Eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Performans metriği hesapla
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Sonuçları:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Skoru: {r2}")


# # Kategorik değişkenleri One-Hot Encoding ile dönüştürelim
# df_encoded = pd.get_dummies(df, columns=['Şehir', 'İlçe', 'Mahalle'], drop_first=True)

# # Bağımsız değişkenler (X) ve bağımlı değişken (y)
# X = df_encoded.drop(columns=["Fiyat (TL)", "Log_Fiyat"])  # Fiyat değişkenlerini çıkarttık
# y = df_encoded["Log_Fiyat"]

# # Eğitim ve test setlerine ayırma
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Modeli oluştur ve eğit
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Tahmin yap
# y_pred = model.predict(X_test)

# # Performans metriği hesapla
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")
# print(f"R² Skoru: {r2}")

#Linear Regression sonrası, ikinci model olarak Decision Tree'yi test edelim:

from sklearn.tree import DecisionTreeRegressor

# Modeli oluştur
dt_model = DecisionTreeRegressor(random_state=42)

# Eğit
dt_model.fit(X_train, y_train)

# Tahmin yap
y_pred_dt = dt_model.predict(X_test)

# Performans metriği hesapla
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regressor Sonuçları:")
print(f"MAE: {mae_dt}")
print(f"MSE: {mse_dt}")
print(f"RMSE: {rmse_dt}")
print(f"R² Skoru: {r2_dt}")



from sklearn.ensemble import RandomForestRegressor

# Random Forest Modeli
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin yap
y_pred_rf = rf_model.predict(X_test)

# Performans metrikleri
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regressor Sonuçları:")
print(f"MAE: {mae_rf}")
print(f"MSE: {mse_rf}")
print(f"RMSE: {rmse_rf}")
print(f"R² Skoru: {r2_rf}")


from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting Modeli
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)

# Tahmin yap
y_pred_gb = gb_model.predict(X_test)

# Performans metrikleri
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Regressor Sonuçları:")
print(f"MAE: {mae_gb}")
print(f"MSE: {mse_gb}")
print(f"RMSE: {rmse_gb}")
print(f"R² Skoru: {r2_gb}")




from xgboost import XGBRegressor

# XGBoost Modeli
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Tahmin yap
y_pred_xgb = xgb_model.predict(X_test)

# Performans metrikleri
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("\nXGBoost Regressor Sonuçları:")
print(f"MAE: {mae_xgb}")
print(f"MSE: {mse_xgb}")
print(f"RMSE: {rmse_xgb}")
print(f"R² Skoru: {r2_xgb}")



from lightgbm import LGBMRegressor

# LightGBM Modeli
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm_model.fit(X_train, y_train)

# Tahmin yap
y_pred_lgbm = lgbm_model.predict(X_test)

# Performans metrikleri
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print("\nLightGBM Regressor Sonuçları:")
print(f"MAE: {mae_lgbm}")
print(f"MSE: {mse_lgbm}")
print(f"RMSE: {rmse_lgbm}")
print(f"R² Skoru: {r2_lgbm}")



from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Veri ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MLP Modeli Tanımlama ve Eğitme
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Tahmin Yapma
y_pred_mlp = mlp_model.predict(X_test_scaled)

# Sonuçları Değerlendirme
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mse_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print("\nMLP Regressor Sonuçları:")
print(f"MAE: {mae_mlp}")
print(f"MSE: {mse_mlp}")
print(f"RMSE: {rmse_mlp}")
print(f"R² Skoru: {r2_mlp}")

from sklearn.ensemble import AdaBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# Eğer GPU varsa kullan, yoksa CPU'yu kullan
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tabnet_model = TabNetRegressor(device_name=device)

# AdaBoost Regressor Modeli
adaboost_model = AdaBoostRegressor(n_estimators=100, random_state=42)
adaboost_model.fit(X_train, y_train)

# Tahmin ve Sonuçları Hesaplama
y_pred_adaboost = adaboost_model.predict(X_test)
mae_adaboost = mean_absolute_error(y_test, y_pred_adaboost)
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
rmse_adaboost = np.sqrt(mse_adaboost)
r2_adaboost = r2_score(y_test, y_pred_adaboost)

print("\nAdaBoost Regressor Sonuçları:")
print(f"MAE: {mae_adaboost}")
print(f"MSE: {mse_adaboost}")
print(f"RMSE: {rmse_adaboost}")
print(f"R² Skoru: {r2_adaboost}")

# TabNet için Veriyi Tensor Formatına Çevirme
X_train_tensor = X_train.to_numpy()
y_train_tensor = y_train.to_numpy().reshape(-1, 1)
X_test_tensor = X_test.to_numpy()
y_test_tensor = y_test.to_numpy().reshape(-1, 1)

# TabNet Modeli
tabnet_model = TabNetRegressor()
tabnet_model.fit(
    X_train_tensor, y_train_tensor,
    eval_set=[(X_test_tensor, y_test_tensor)],
    max_epochs=100, patience=10,
    batch_size=1024, virtual_batch_size=128
)

# TabNet Tahmin ve Sonuçları
y_pred_tabnet = tabnet_model.predict(X_test_tensor).flatten()
mae_tabnet = mean_absolute_error(y_test_tensor, y_pred_tabnet)
mse_tabnet = mean_squared_error(y_test_tensor, y_pred_tabnet)
rmse_tabnet = np.sqrt(mse_tabnet)
r2_tabnet = r2_score(y_test_tensor, y_pred_tabnet)

print("\nTabNet Regressor Sonuçları:")
print(f"MAE: {mae_tabnet}")
print(f"MSE: {mse_tabnet}")
print(f"RMSE: {rmse_tabnet}")
print(f"R² Skoru: {r2_tabnet}")



from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Modeli oluştur ve eğit
catboost = CatBoostRegressor(verbose=100, iterations=500, learning_rate=0.1, depth=6)
catboost.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

# Tahmin yap
y_pred_catboost = catboost.predict(X_test)

# Performans metrikleri
mae = mean_absolute_error(y_test, y_pred_catboost)
mse = mean_squared_error(y_test, y_pred_catboost)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_catboost)

print("CatBoost Regressor Sonuçları:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R² Skoru: {r2}")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_catboost, alpha=0.5, label="CatBoost")
plt.scatter(y_test, y_pred_lgbm, alpha=0.5, label="LightGBM")
plt.scatter(y_test, y_pred_xgb, alpha=0.5, label="XGBoost")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--") # İdeal çizgi
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.legend()
plt.title("Gerçek vs. Tahmin Edilen Değerler")
plt.show()

sns.histplot(y_test - y_pred_catboost, bins=50, kde=True, color="blue", label="CatBoost")
sns.histplot(y_test - y_pred_lgbm, bins=50, kde=True, color="green", label="LightGBM")
plt.legend()
plt.xlabel("Tahmin Hatası (Residuals)")
plt.title("Model Hata Dağılımı")
plt.show()

# Konfüzyon matrisi (confusion matrix) deneme

# Tahmin ve gerçek değerler arasındaki farkları hesapla
errors = y_test - y_pred  # Gerçek - Tahmin

# Hataların histogramını çiz
plt.figure(figsize=(10, 5))
sns.histplot(errors, bins=50, kde=True, color='red')
plt.xlabel("Hata (Gerçek - Tahmin)")
plt.ylabel("Frekans")
plt.title("Hata Dağılımı")
plt.show()

# Gerçek vs Tahmin edilen değerleri karşılaştıran bir scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Doğru tahminler için y=x çizgisi
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gerçek ve Tahmin Edilen Fiyatlar Karşılaştırması")
plt.show()

# Sonuçları değerlendirme
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Sonuçları:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Skoru: {r2}")