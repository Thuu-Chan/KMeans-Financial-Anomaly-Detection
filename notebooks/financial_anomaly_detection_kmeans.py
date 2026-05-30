# =========================
# BƯỚC 1: IMPORT THƯ VIỆN
# =========================
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import kruskal
from scipy.stats import f_oneway
import matplotlib.pyplot as plt
os.makedirs("results", exist_ok=True)
import seaborn as sns

# =========================
# GLOBAL VISUAL SETTINGS
# =========================
sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11
})

# =========================
# BƯỚC 2: ĐỌC & LÀM SẠCH DỮ LIỆU
# =========================
df = pd.read_excel("/content/160maudoanhnghiep.xlsx")

df.columns = df.columns.str.strip()

features = [
    'ROA',
    'Revenue Growth',
    'Accruals',
    'Tax Ratio',
    'Receivable Ratio'
]

df_model = df.copy()

# chuyển về numeric
for col in features:
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

# loại bỏ missing
df_model = df_model.dropna(subset=features)

print("Số dòng dữ liệu:", df_model.shape)
# =========================
# BƯỚC 2.1: BIẾN ĐỔI LOGIC NGHIỆP VỤ (TRỊ TUYỆT ĐỐI)
# =========================
# Lấy trị tuyệt đối cho các biến mà sự sai lệch (cả âm và dương) đều là bất thường
abs_features = ['Accruals', 'Revenue Growth']

for col in abs_features:
    df_model[col] = df_model[col].abs()

print("Đã lấy trị tuyệt đối cho:", abs_features)
# =========================
# BƯỚC 3: THỐNG KÊ MÔ TẢ 
# =========================
desc_stats = df_model[features].describe().T

# thêm median + quartiles cho bài báo
desc_stats['median'] = df_model[features].median()
desc_stats['q1'] = df_model[features].quantile(0.25)
desc_stats['q3'] = df_model[features].quantile(0.75)

# sắp xếp lại cột
desc_stats = desc_stats[['mean', 'std', 'min', 'q1', 'median', 'q3', 'max']]

print(desc_stats)

desc_stats.to_excel(
    "results/DESCRIPTIVE_STATISTICS_FULL.xlsx"
)

# =========================
# BƯỚC 3.1: BOXPLOT (NHẬN DIỆN OUTLIERS)
# =========================
plt.figure(figsize=(10, 6))

sns.boxplot(data=df_model[features])

plt.title("Boxplot of Financial Variables")
plt.xticks(rotation=45)
plt.savefig(
    "results/boxplot.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# =========================
# BƯỚC 4: CHUẨN HÓA DỮ LIỆU
# =========================
scaler = RobustScaler()

X_scaled = scaler.fit_transform(df_model[features])

# =========================
# BƯỚC 5: CHỌN K TỐI ƯU (SILHOUETTE)
# =========================
K_range = range(2, 7)

scores = []

for k in K_range:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    scores.append(score)

    print(f"K = {k}, Score = {score:.4f}")

# =========================
# BƯỚC 5.1: CHỌN K THEO KẾT HỢP 
# =========================
best_k = 3  # chọn theo ý nghĩa kinh tế

print("Silhouette Scores:", scores)
print("Chọn K =", best_k, "(gần tối ưu và có ý nghĩa kinh tế)")

# vẽ biểu đồ chọn K
plt.figure(figsize=(6, 4))

plt.plot(K_range, scores, marker='o')

plt.title("Selecting Optimal K using Silhouette Score")
plt.xlabel("K")
plt.ylabel("Score")

plt.grid(True)

plt.show()

# =========================
# STEP 5.2: SILHOUETTE PLOT ANALYSIS
# (Phân tích cấu trúc cụm chi tiết bằng biểu đồ Silhouette)
# =========================
from sklearn.metrics import silhouette_samples

import numpy as np

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X_scaled)

sample_silhouette_values = silhouette_samples(
    X_scaled,
    labels
)

plt.figure(figsize=(8, 6))

y_lower = 10

for i in range(3):

    ith_cluster_silhouette_values = (
        sample_silhouette_values[labels == i]
    )

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]

    y_upper = y_lower + size_cluster_i

    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values
    )

    plt.text(
        -0.05,
        y_lower + 0.5 * size_cluster_i,
        str(i)
    )

    y_lower = y_upper + 10

plt.axvline(
    x=np.mean(sample_silhouette_values),
    linestyle="--"
)

plt.title(
    "Silhouette Analysis for K = 3",
    weight='bold',
    pad=15
)

plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")

plt.grid(False)

plt.savefig(
    "results/silhouette_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# BƯỚC 6: PHÂN CỤM KMEANS
# =========================
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

df_model['cluster'] = kmeans.fit_predict(X_scaled)

print("Các cluster:", df_model['cluster'].value_counts())

# =========================
# Bước 7
df_model['anomaly_score'] = (
    0.30 * df_model['Accruals']  # Đã là trị tuyệt đối từ Bước 2
    + 0.25 * df_model['Receivable Ratio']
    + 0.20 * (1 - df_model['Tax Ratio'])
    + 0.15 * df_model['Revenue Growth'] # Đã là trị tuyệt đối từ Bước 2
    + 0.10 * (1 - df_model['ROA'])
)
# =========================
# BƯỚC 8: GÁN NHÃN RỦI RO
# =========================
cluster_mean = (
    df_model
    .groupby('cluster')['anomaly_score']
    .mean()
    .sort_values()
)

labels = {
    cluster_mean.index[0]: "Normal",
    cluster_mean.index[1]: "Watch",
    cluster_mean.index[2]: "High_Risk"
}

df_model['Risk_Level'] = df_model['cluster'].map(labels)

# bảng màu cố định
risk_colors = {
    "Normal": "green",
    "Watch": "orange",
    "High_Risk": "red"
}

# =========================
# BƯỚC 9: XUẤT KẾT QUẢ
# =========================
result = df_model[
    [
        'Firm_Name',
        'Ticker',
        'Year',
        'Risk_Level',
        'anomaly_score'
    ]
]

result.to_excel(
    "results/KET_QUA_KMEANS.xlsx",
    index=False
)

# =========================
# BƯỚC 10: PCA VISUALIZATION
# =========================
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

df_model['pca1'] = X_pca[:, 0]
df_model['pca2'] = X_pca[:, 1]

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df_model,
    x='pca1',
    y='pca2',
    hue='Risk_Level',
    palette=risk_colors,
    hue_order=["Normal", "Watch", "High_Risk"],
    s=100,
    edgecolor='black'
)

plt.title(
    "PCA Scatter Plot of Firms by Risk Group",
    pad=15,
    weight='bold'
)

plt.xlabel("Principal Component 1 (PC1)")
plt.ylabel("Principal Component 2 (PC2)")

plt.legend(title="Risk Level")

plt.savefig(
    "results/pca_scatter.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# BƯỚC 11: HEATMAP ĐẶC ĐIỂM CỤM
# =========================
cluster_profile = (
    df_model
    .groupby('Risk_Level')[features]
    .mean()
)

print(cluster_profile)

cluster_profile.to_excel(
    "results/cluster_profile.xlsx"
)

plt.figure(figsize=(8, 5))

sns.heatmap(
    cluster_profile,
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Financial Characteristics by Risk Group",
    pad=15,
    weight='bold'
)

plt.xlabel("Financial Indicators")
plt.ylabel("Risk Group")

plt.savefig(
    "results/cluster_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# BƯỚC 11.1: LINE CHART
# =========================
# STEP 11.1: LINE CHART - FINANCIAL PROFILE BY RISK GROUP
# =========================
plt.figure(figsize=(9, 5))

for risk in ["Normal", "Watch", "High_Risk"]:

    plt.plot(
        cluster_profile.columns,
        cluster_profile.loc[risk],
        marker='o',
        linewidth=2.5,
        label=risk,
        color=risk_colors[risk]  # 🔥 dùng bảng màu
    )

plt.title(
    "Financial Profile by Risk Group",
    pad=15,
    weight='bold'
)

plt.xlabel("Financial Indicators")
plt.ylabel("Average Value")

plt.legend(title="Risk Group")

plt.grid(
    True,
    linestyle='--',
    alpha=0.6
)

plt.tight_layout()

plt.savefig(
    "results/financial_profile.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# =========================
# BƯỚC 12: PHÂN PHỐI ANOMALY SCORE (LOG SCALE)
# =========================
import matplotlib.ticker as ticker

plt.figure(figsize=(8, 5))

# Vẽ biểu đồ với màu sắc đậm hơn để không bị "nhạt"
sns.histplot(
    df_model['anomaly_score'],
    bins=40,
    kde=True,
    color='#2b5797',  # Màu xanh đậm chuyên nghiệp
    edgecolor='black',
    alpha=0.7
)

# Chuyển trục Y sang dạng Logarit để hiển thị rõ "đuôi nặng"
plt.yscale('log')

# Định dạng lại nhãn trục Y để dễ đọc (thay vì dạng số mũ 10^k)
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())

plt.title("Distribution of Anomaly Scores (Logarithmic Scale)", pad=15, weight='bold')
plt.xlabel("Anomaly Score (S)")
plt.ylabel("Frequency (Log Scale)")

# Thêm lưới để dễ đối chiếu các điểm ở đuôi
plt.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "results/anomaly_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# BƯỚC 13: SỐ LƯỢNG THEO RISK
# =========================
df_model['Risk_Level'].value_counts().plot(
    kind='bar',
    figsize=(6, 4)
)

plt.title("Number of Firms by Risk Level")

plt.xlabel("Risk Level")
plt.ylabel("Count")

plt.savefig(
    "results/risk_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# BƯỚC 14: KIỂM ĐỊNH KRUSKAL-WALLIS (QUAN TRỌNG)
# =========================
print("\n=== KRUSKAL-WALLIS TEST ===")

for col in features:

    groups = [
        df_model[df_model['Risk_Level'] == level][col]
        for level in ["Normal", "Watch", "High_Risk"]
    ]

    stat, p = kruskal(*groups)

    print(f"{col}: p-value = {p:.4f}")

# =========================
# BƯỚC 15: KIỂM ĐỊNH ANOVA (BỔ SUNG)
# =========================
from scipy.stats import f_oneway

print("\n=== ANOVA TEST ===")

for col in features:

    groups = [
        df_model[df_model['Risk_Level'] == level][col]
        for level in ["Normal", "Watch", "High_Risk"]
    ]

    stat, p = f_oneway(*groups)

    print(f"{col}: F-stat = {stat:.4f}, p-value = {p:.4f}")

anova_results = []

for col in features:

    groups = [
        df_model[df_model['Risk_Level'] == level][col]
        for level in ["Normal", "Watch", "High_Risk"]
    ]

    stat, p = f_oneway(*groups)

    anova_results.append({
        "Variable": col,
        "F-stat": stat,
        "p-value": p
    })

anova_df = pd.DataFrame(anova_results)

anova_df.to_excel(
    "results/ANOVA_RESULTS.xlsx",
    index=False
)

print(anova_df)
