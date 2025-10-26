# Fáza 1 - Prieskumná analýza dát
# Critical Oxygen Saturation Estimation
# Percentuálny podiel práce: [Meno1: X%, Meno2: Y%]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene
import warnings

warnings.filterwarnings('ignore')

# Nastavenie štýlu pre vizualizácie
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)



print("=" * 80)
print("1.1 ZÁKLADNÝ OPIS DÁT")
print("=" * 80)

# Načítanie dát
df = pd.read_csv('Datasets/observation.csv', sep='\t')

# (A-1b) Analýza štruktúr dát
print("\n--- (A) ANALÝZA ŠTRUKTÚRY DÁT ---")
print(f"Počet záznamov: {df.shape[0]}")
print(f"Počet atribútov: {df.shape[1]}")
print(f"\nTypy atribútov:\n{df.dtypes.value_counts()}")
print(f"\nNázvy stĺpcov:\n{df.columns.tolist()}")
print(f"\nPrvých 5 záznamov:")
print(df.head())

print("\n\nDetailné informácie o dátach:")
df.info()

print("\n\nPamäťová náročnosť:")
print(f"Celková veľkosť: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# (B-1b) Analýza jednotlivých atribútov (min 10)
print("\n\n--- (B) ANALÝZA JEDNOTLIVÝCH ATRIBÚTOV ---")

# Vyberieme významné atribúty pre analýzu
significant_attrs = ['SpO₂', 'Blood Flow Index', 'PPG waveform features',
                     'Signal Quality Index', 'Respiratory effort',
                     'O₂ extraction ratio', 'SNR', 'oximetry',
                     'latitude', 'longitude', 'BP', 'RR', 'HR']

# Skontrolujeme ktoré atribúty existujú
available_attrs = [col for col in significant_attrs if col in df.columns]
print(f"\nAnalyzované atribúty ({len(available_attrs)}): {available_attrs}")

# Deskriptívne štatistiky
print("\n\nDESKRIPTÍVNE ŠTATISTIKY:")
print(df[available_attrs].describe())

# Kontrola rozsahov hodnôt pre fyziologické parametre
print("\n\nKONTROLA ROZSAHOV HODNÔT:")

# SpO₂ by malo byť 0-100%
if 'SpO₂' in df.columns:
    invalid_spo2 = df[(df['SpO₂'] < 0) | (df['SpO₂'] > 100)]
    print(f"SpO₂: rozsah [{df['SpO₂'].min():.2f}, {df['SpO₂'].max():.2f}]")
    print(f"  - Neplatné hodnoty (mimo 0-100): {len(invalid_spo2)}")

# Oximetry (target variable)
if 'oximetry' in df.columns:
    print(f"Oximetry: rozsah [{df['oximetry'].min():.2f}, {df['oximetry'].max():.2f}]")
    print(f"  - Jedinečných hodnôt: {df['oximetry'].nunique()}")

# Distribúcie atribútov - vizualizácia
print("\n\nVIZUALIZÁCIA DISTRIBÚCIÍ:")
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.ravel()

for idx, col in enumerate(available_attrs[:12]):
    if col in df.columns:
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribúcia: {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frekvencia')

        # Pridanie štatistík
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
        axes[idx].legend(fontsize=8)

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Boxploty pre identifikáciu outlierov
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
axes = axes.ravel()

for idx, col in enumerate(available_attrs[:12]):
    if col in df.columns:
        axes[idx].boxplot(df[col].dropna(), vert=True)
        axes[idx].set_title(f'Boxplot: {col}')
        axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# (C-1b) Párová analýza dát - vzťahy medzi atribútmi
print("\n\n--- (C) PÁROVÁ ANALÝZA - VZŤAHY MEDZI ATRIBÚTMI ---")

# Korelačná matica
numeric_cols = df[available_attrs].select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

print("\nKORELAČNÁ MATICA:")
print(correlation_matrix)

# Vizualizácia korelačnej matice
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Korelačná matica atribútov', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Identifikácia silných korelácií (|r| > 0.7)
print("\n\nSILNÉ KORELÁCIE (|r| > 0.7):")
strong_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            strong_corr.append({
                'Atribút 1': correlation_matrix.columns[i],
                'Atribút 2': correlation_matrix.columns[j],
                'Korelácia': correlation_matrix.iloc[i, j]
            })

if strong_corr:
    print(pd.DataFrame(strong_corr))
else:
    print("Žiadne silné korelácie medzi atribútmi.")

# (D-1b) Párová analýza - závislosti s predikovanou premennou
print("\n\n--- (D) ZÁVISLOSTI S PREDIKOVANOU PREMENNOU (oximetry) ---")

if 'oximetry' in df.columns:
    target_correlations = correlation_matrix['oximetry'].sort_values(ascending=False)
    print("\nKORELÁCIE S OXIMETRY:")
    print(target_correlations)

    # Vizualizácia top korelácií
    plt.figure(figsize=(12, 8))
    target_correlations.drop('oximetry').plot(kind='barh', color='steelblue')
    plt.title('Korelácie atribútov s oximetry', fontsize=14)
    plt.xlabel('Korelačný koeficient')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('target_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Scatter ploty pre top 6 najsilnejších prediktorov
    top_predictors = target_correlations.drop('oximetry').abs().nlargest(6).index

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    for idx, predictor in enumerate(top_predictors):
        axes[idx].scatter(df[predictor], df['oximetry'], alpha=0.5)
        axes[idx].set_xlabel(predictor)
        axes[idx].set_ylabel('oximetry')
        axes[idx].set_title(f'{predictor} vs oximetry (r={target_correlations[predictor]:.3f})')

        # Pridanie regresnej línie
        z = np.polyfit(df[predictor].dropna(), df.loc[df[predictor].notna(), 'oximetry'], 1)
        p = np.poly1d(z)
        axes[idx].plot(df[predictor].sort_values(), p(df[predictor].sort_values()),
                       "r--", alpha=0.8, linewidth=2)

    plt.tight_layout()
    plt.savefig('predictor_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()

# (E-1b) Prvotné zamyslenie k riešeniu projektu
print("\n\n--- (E) PRVOTNÉ ZAMYSLENIE K RIEŠENIU ---")

# ============================================================================
# 1.2 IDENTIFIKÁCIA PROBLÉMOV, INTEGRÁCIA A ČISTENIE DÁT (5b)
# ============================================================================

print("\n\n" + "=" * 80)
print("1.2 IDENTIFIKÁCIA PROBLÉMOV, INTEGRÁCIA A ČISTENIE DÁT")
print("=" * 80)

# (A-2b) Identifikácia problémov v dátach
print("\n--- (A) IDENTIFIKÁCIA PROBLÉMOV V DÁTACH ---")

# 1. Chýbajúce hodnoty
print("\n1. CHÝBAJÚCE HODNOTY:")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Atribút': missing_data.index,
    'Počet chýbajúcich': missing_data.values,
    'Percento': missing_percent.values
}).sort_values('Počet chýbajúcich', ascending=False)

print(missing_df[missing_df['Počet chýbajúcich'] > 0])

# Vizualizácia chýbajúcich hodnôt
plt.figure(figsize=(12, 8))
missing_df_plot = missing_df[missing_df['Počet chýbajúcich'] > 0]
if not missing_df_plot.empty:
    plt.barh(missing_df_plot['Atribút'], missing_df_plot['Percento'])
    plt.xlabel('Percento chýbajúcich hodnôt (%)')
    plt.title('Chýbajúce hodnoty v dátach')
    plt.tight_layout()
    plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. Duplicitné záznamy
print("\n2. DUPLICITNÉ ZÁZNAMY:")
duplicates = df.duplicated().sum()
print(f"Počet duplicitných záznamov: {duplicates}")
if duplicates > 0:
    print("Príklad duplicitných záznamov:")
    print(df[df.duplicated(keep=False)].head())

# 3. Nejednotné formáty
print("\n3. KONTROLA FORMÁTOV:")
for col in df.columns:
    unique_types = df[col].apply(type).unique()
    if len(unique_types) > 1:
        print(f"Atribút '{col}' má nejednotné typy: {unique_types}")

# 4. Prázdne stringy a whitespace
print("\n4. PRÁZDNE HODNOTY A WHITESPACE:")
for col in df.select_dtypes(include=['object']).columns:
    empty_strings = (df[col] == '').sum()
    whitespace = df[col].str.strip().eq('').sum() if df[col].dtype == 'object' else 0
    if empty_strings > 0 or whitespace > 0:
        print(f"{col}: {empty_strings} prázdnych stringov, {whitespace} whitespace")

# (B-2b) Kontrola správnosti v dátach
print("\n\n--- (B) KONTROLA SPRÁVNOSTI V DÁTACH ---")

# 1. Abnormálne hodnoty
print("\n1. ABNORMÁLNE HODNOTY:")

# SpO₂ kontrola
if 'SpO₂' in df.columns:
    abnormal_spo2 = df[(df['SpO₂'] < 0) | (df['SpO₂'] > 100)]
    print(f"SpO₂ mimo rozsahu [0, 100]: {len(abnormal_spo2)} záznamov")
    if len(abnormal_spo2) > 0:
        print(f"  Min: {df['SpO₂'].min()}, Max: {df['SpO₂'].max()}")

# Kontrola negatívnych hodnôt tam, kde by nemali byť
print("\nNegatívne hodnoty v atribútoch:")
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"  {col}: {negative_count} negatívnych hodnôt")

# 2. Nelogické dátové vzťahy
print("\n2. NELOGICKÉ DÁTOVÉ VZŤAHY:")
# Príklad: ak existujú časové stĺpce, kontrola časovej konzistencie
# Príklad: kontrola geografických súradníc
if 'latitude' in df.columns and 'longitude' in df.columns:
    invalid_coords = df[(df['latitude'].abs() > 90) | (df['longitude'].abs() > 180)]
    print(f"Neplatné geografické súradnice: {len(invalid_coords)} záznamov")

# (C-1b) Vychýlené hodnoty - Outlier Detection
print("\n\n--- (C) DETEKCIA A RIEŠENIE OUTLIEROV ---")

# Technika 1: IQR metóda
print("\n1. TECHNIKA: IQR METÓDA")


def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


outlier_summary = []
for col in numeric_cols[:5]:  # Analyzujeme prvých 5 numerických stĺpcov
    outliers, lower, upper = detect_outliers_iqr(df, col)
    outlier_summary.append({
        'Atribút': col,
        'Počet outlierov': len(outliers),
        'Percento': (len(outliers) / len(df)) * 100,
        'Dolná hranica': lower,
        'Horná hranica': upper
    })

print(pd.DataFrame(outlier_summary))

# Technika 2: Z-score metóda
print("\n2. TECHNIKA: Z-SCORE METÓDA (|z| > 3)")


def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(stats.zscore(data[column].dropna()))
    outliers = data[column][z_scores > threshold]
    return len(outliers)


zscore_summary = []
for col in numeric_cols[:5]:
    outlier_count = detect_outliers_zscore(df, col)
    zscore_summary.append({
        'Atribút': col,
        'Počet outlierov': outlier_count,
        'Percento': (outlier_count / len(df)) * 100
    })

print(pd.DataFrame(zscore_summary))

# Riešenie outlierov - príklad
print("\n\nRIEŠENIE OUTLIEROV:")
df_cleaned = df.copy()

# Metóda 1: Odstránenie outlierov (príklad pre jeden atribút)
if 'SpO₂' in df.columns:
    outliers, lower, upper = detect_outliers_iqr(df, 'SpO₂')
    df_no_outliers = df[(df['SpO₂'] >= lower) & (df['SpO₂'] <= upper)]
    print(f"\nMetóda 1 - Odstránenie (SpO₂):")
    print(f"  Pôvodný počet: {len(df)}")
    print(f"  Po odstránení outlierov: {len(df_no_outliers)}")
    print(f"  Odstránených: {len(df) - len(df_no_outliers)}")

# Metóda 2: Winsorization - nahradenie hraničnými hodnotami
print(f"\nMetóda 2 - Winsorization (5% a 95% percentil):")
for col in numeric_cols[:3]:
    p5 = df[col].quantile(0.05)
    p95 = df[col].quantile(0.95)
    before_count = ((df[col] < p5) | (df[col] > p95)).sum()
    df_cleaned[col] = df[col].clip(lower=p5, upper=p95)
    print(f"  {col}: {before_count} hodnôt upravených")

# ============================================================================
# 1.3 FORMULÁCIA A ŠTATISTICKÉ OVERENIE HYPOTÉZ (5b)
# ============================================================================

print("\n\n" + "=" * 80)
print("1.3 FORMULÁCIA A ŠTATISTICKÉ OVERENIE HYPOTÉZ")
print("=" * 80)

# (A-4b) Formulácia a overenie hypotéz
print("\n--- (A) FORMULÁCIA A OVERENIE HYPOTÉZ ---")

# HYPOTÉZA 1
print("\n" + "=" * 60)
print("HYPOTÉZA 1")
print("=" * 60)
print("H0: SpO₂ má rovnakú priemernú hodnotu pre vysoké a nízke oximetry")
print("H1: SpO₂ má vyššiu priemernú hodnotu pri vysokom oximetry")

if 'SpO₂' in df.columns and 'oximetry' in df.columns:
    # Rozdelenie na dve skupiny podľa mediánu oximetry
    median_oximetry = df['oximetry'].median()
    high_oximetry = df[df['oximetry'] >= median_oximetry]['SpO₂'].dropna()
    low_oximetry = df[df['oximetry'] < median_oximetry]['SpO₂'].dropna()

    print(f"\nPopisné štatistiky:")
    print(
        f"Vysoké oximetry - SpO₂: mean={high_oximetry.mean():.2f}, std={high_oximetry.std():.2f}, n={len(high_oximetry)}")
    print(f"Nízke oximetry - SpO₂: mean={low_oximetry.mean():.2f}, std={low_oximetry.std():.2f}, n={len(low_oximetry)}")

    # Test normality (Shapiro-Wilk)
    print("\nTest normality (Shapiro-Wilk):")
    _, p_high = shapiro(high_oximetry.sample(min(5000, len(high_oximetry))))
    _, p_low = shapiro(low_oximetry.sample(min(5000, len(low_oximetry))))
    print(f"  Vysoké oximetry: p-value = {p_high:.4f}")
    print(f"  Nízke oximetry: p-value = {p_low:.4f}")

    # Test homogenity rozptylov (Levene)
    _, p_levene = levene(high_oximetry, low_oximetry)
    print(f"\nLevene test (homogenita rozptylov): p-value = {p_levene:.4f}")

    # Výber testu
    if p_high > 0.05 and p_low > 0.05 and p_levene > 0.05:
        print("\nPoužitý test: t-test (parametrický)")
        t_stat, p_value = ttest_ind(high_oximetry, low_oximetry)
        test_name = "t-test"
    else:
        print("\nPoužitý test: Mann-Whitney U (neparametrický)")
        t_stat, p_value = mannwhitneyu(high_oximetry, low_oximetry, alternative='greater')
        test_name = "Mann-Whitney U"

    print(f"\nVýsledky {test_name}:")
    print(f"  Test statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Záver: {'Zamietame H0' if p_value < 0.05 else 'Nezamietame H0'} (α=0.05)")

    # Effect size (Cohen's d)
    cohens_d = (high_oximetry.mean() - low_oximetry.mean()) / np.sqrt(
        ((len(high_oximetry) - 1) * high_oximetry.std() ** 2 + (len(low_oximetry) - 1) * low_oximetry.std() ** 2) / (
                    len(high_oximetry) + len(low_oximetry) - 2))
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")

    # Vizualizácia
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(high_oximetry, bins=30, alpha=0.5, label='Vysoké oximetry', color='green')
    plt.hist(low_oximetry, bins=30, alpha=0.5, label='Nízke oximetry', color='red')
    plt.xlabel('SpO₂')
    plt.ylabel('Frekvencia')
    plt.legend()
    plt.title('Hypotéza 1: Distribúcia SpO₂')

    plt.subplot(1, 2, 2)
    plt.boxplot([high_oximetry, low_oximetry], labels=['Vysoké oximetry', 'Nízke oximetry'])
    plt.ylabel('SpO₂')
    plt.title('Porovnanie skupín')
    plt.tight_layout()
    plt.savefig('hypothesis1.png', dpi=300, bbox_inches='tight')
    plt.show()

# HYPOTÉZA 2
print("\n\n" + "=" * 60)
print("HYPOTÉZA 2")
print("=" * 60)
print("H0: Signal Quality Index nemá vplyv na presnosť oximetry")
print("H1: Vyšší Signal Quality Index vedie k vyššej oximetry")

if 'Signal Quality Index' in df.columns and 'oximetry' in df.columns:
    # Výpočet korelácie
    corr, p_corr = stats.pearsonr(df['Signal Quality Index'].dropna(),
                                  df.loc[df['Signal Quality Index'].notna(), 'oximetry'])

    print(f"\nPearsonova korelácia:")
    print(f"  r = {corr:.4f}")
    print(f"  p-value = {p_corr:.4f}")
    print(f"  Záver: {'Zamietame H0' if p_corr < 0.05 else 'Nezamietame H0'} (α=0.05)")

    # Spearman korelácia (non-parametric alternative)
    corr_s, p_corr_s = stats.spearmanr(df['Signal Quality Index'].dropna(),
                                       df.loc[df['Signal Quality Index'].notna(), 'oximetry'])
    print(f"\nSpearmanova korelácia:")
    print(f"  ρ = {corr_s:.4f}")
    print(f"  p-value = {p_corr_s:.4f}")

    # Vizualizácia
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Signal Quality Index'], df['oximetry'], alpha=0.3)
    plt.xlabel('Signal Quality Index')
    plt.ylabel('Oximetry')
    plt.title(f'Hypotéza 2: Signal Quality Index vs Oximetry (r={corr:.3f}, p={p_corr:.4f})')

    # Regresná línia
    z = np.polyfit(df['Signal Quality Index'].dropna(),
                   df.loc[df['Signal Quality Index'].notna(), 'oximetry'], 1)
    p = np.poly1d(z)
    plt.plot(df['Signal Quality Index'].sort_values(),
             p(df['Signal Quality Index'].sort_values()), "r--", linewidth=2)

    plt.tight_layout()
    plt.savefig('hypothesis2.png', dpi=300, bbox_inches='tight')
    plt.show()

# (B-1b) Štatistická sila testov
print("\n\n--- (B) ŠTATISTICKÁ SILA TESTOV ---")

from statsmodels.stats.power import ttest_power

print("\nANALÝZA ŠTATISTICKEJ SILY:")

# Pre Hypotézu 1
if 'SpO₂' in df.columns and 'oximetry' in df.columns:
    effect_size = abs(cohens_d)
    sample_size = min(len(high_oximetry), len(low_oximetry))
    power = ttest_power(effect_size, sample_size, alpha=0.05, alternative='two-sided')

    print(f"\nHypotéza 1:")
    print(f"  Effect size (Cohen's d): {effect_size:.4f}")
    print(f"  Veľkosť vzorky: {sample_size}")
    print(f"  Štatistická sila (power): {power:.4f}")
    print(f"  Interpretácia: {'Dostatočná' if power >= 0.8 else 'Nedostatočná'} (požadované ≥ 0.8)")

# Výpočet potrebnej veľkosti vzorky pre power = 0.8
from statsmodels.stats.power import tt_solve_power

required_n = tt_solve_power(effect_size=0.5, alpha=0.05, power=0.8, alternative='two-sided')
print(f"  Požadovaná veľkosť vzorky pre power=0.8: {int(required_n)} na skupinu")

print("\n\n" + "=" * 80)












detected_outliers = detect_outliers(observations, "PVI")
outliers_perc = (detected_outliers.sum()/observations["PVI"].count()) * 100
print(f"{outliers_perc:.2f}%")

print(observations[~detected_outliers]["PVI"].mean())
print(observations[~detected_outliers]["PVI"].median())
print(observations[~detected_outliers]["PVI"].var())





mask_n = (observations["SpO₂"] > spo2_limit) & (observations["FiO₂"] < fio2_limit)
mask_n = (observations["SV"] > sv_limit_h) & (observations["CO"] < co_limit)   
mask_n = (observations["Respiratory effort"] > re_limit) & (observations["Blood Flow Index"] < bfi_limit)
mask_n = (observations["PRV"] > prv_limit) & (observations["Blood Flow Index"] < bfi_limit)
mask_n = (observations["Skin Temperature"] > st_limit_h) & (observations["HR"] < hr_limit_l)
mask_n = (observations["Skin Temperature"] > st_limit_l) & (observations["Blood Flow Index"] < bfi_limit)
mask_n = (observations["RR"] > rr_limit) & (observations["HR"] < hr_limit)
mask_n = (observations["PVI"] > pvi_limit) & (observations["SV"] > sv_limit_l)
mask_n = (observations["O₂ extraction ratio"] > o2er_limit) & (observations["CO"] < co_limit)
