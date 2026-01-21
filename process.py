import pandas as pd
import numpy as np
from pathlib import Path
import openpyxl 

# DOSYA YOLLARI
IN = Path("data.xlsx")
OUT_XLSX = Path("data_output.xlsx")

df = pd.read_excel(IN)
df["ds"] = pd.to_datetime(df["ds"], dayfirst=True)


if "EDU" in df.columns:
    df["EDU"] = df["EDU"].fillna(0.0)

# AY
# 
df["month"] = df["ds"].dt.month
 
# eğer veride outlier değerler analiz edilip Ozel_Durum kolonu oluşturulduysa, 1 olanları at
#2. Özel Durum Filtresi (1 Olanları At)
if "Ozel_Durum" in df.columns:
    df["Ozel_Durum"] = pd.to_numeric(df["Ozel_Durum"], errors='coerce').fillna(0).astype(int).astype(str)
    print(f"Filtreleme Öncesi: {len(df)}")
    df = df[df["Ozel_Durum"] != "1"]
    print(f"Filtreleme Sonrası: {len(df)}")
else:
    print("UYARI: Ozel_Durum bulunamadı.")

df['month'] = df['ds'].dt.month



# AGGREGATION 
monthly = (
    df
    .groupby(['ds', 'SKU_GROUP_ID_DESC', 'DEPARTMENT_ID_DESC', 'month', 'Ozel_Durum'], as_index=False)
    .agg({
        "LEGAL_PRICE": "sum",
        "BUDGET_PRICE": "sum",
        "eur_try": "mean",
        "Enflasyon": "mean",
        "EDU": "mean"  # 'first' yerine 'mean' kullanıldı
    })
)

# RENAME 
monthly = monthly.rename(columns={
    "LEGAL_PRICE": "y",
    "BUDGET_PRICE": "budget_month",
    "eur_try": "eurtry_month",
    "Enflasyon": "inflation",
    "EDU": "edu"
})

# Time Index Hesaplama
min_date = monthly["ds"].min()
monthly["time_idx"] = (
    (monthly["ds"].dt.year - min_date.year) * 12 +
    (monthly["ds"].dt.month - min_date.month)
).astype(int)

# month_sin ve month_cos Hesaplama
monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

cols_to_fix = ["legal_price", "budget_price", "edu"]

for col in cols_to_fix:
    if col in monthly.columns:
        # 1. Önce sayısal formata zorla (hatalı stringleri NaN yapar)
        monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
        
        # 2. NaN olanları 0 ile doldur
        monthly[col] = monthly[col].fillna(0.0)
        
        # 3. Kafa karışıklığını önlemek için 2 basamağa yuvarlanabiir
        # (Model için şart değil ama veriyi incelediğinizde temiz görünür)
        # monthly[col] = monthly[col].round(2) 
        
        # 4. Veri tipini float32 yap (TFT modelleri float32 sever, hafıza dostudur)
        monthly[col] = monthly[col].astype("float32")

print("✅ Fiyat ve EDU kolonları float32 formatına dönüştürüldü.")

# Sonuçları Kaydetme 
monthly.to_excel(OUT_XLSX, index=False)

print("✅ EDU verisi normalize edildi ve aggregation yapısı düzeltildi.")
print("✅ month_sin ve month_cos hesaplandı.")
print("✅ time_idx düzgün hesaplandı.")