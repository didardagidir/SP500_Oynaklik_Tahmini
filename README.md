# S&P 500 Gerçekleşen Oynaklığının Optimizasyon Tabanlı ML Modelleri ve Açıklanabilir Yapay Zekâ ile Tahmini

S&P 500 endeksinin **5 gün ileri gerçekleşen oynaklığını (realized volatility)** tahmin etmek için zaman serisi problemi denetimli bir regresyon problemine dönüştürülmüş; beş model (klasik GARCH baseline dahil) üç farklı optimizasyon stratejisiyle eğitilmiş ve en iyi model üç bağımsız XAI yöntemiyle yorumlanmıştır.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-GBM-success)
![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20Opt-purple)
![SHAP](https://img.shields.io/badge/SHAP-XAI-red)

---

## İçindekiler

- [Motivasyon](#motivasyon)
- [Problem Tanımı](#problem-tanımı)
- [Veri Seti](#veri-seti)
- [Öznitelik Mühendisliği](#öznitelik-mühendisliği)
- [Yöntem](#yöntem)
- [Sonuçlar](#sonuçlar)
- [Açıklanabilir Yapay Zekâ (XAI)](#açıklanabilir-yapay-zekâ-xai)
- [Kilit Bulgular](#kilit-bulgular)
- [Proje Yapısı](#proje-yapısı)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sınırlılıklar ve Gelecek Çalışma](#sınırlılıklar-ve-gelecek-çalışma)
- [Kaynaklar](#kaynaklar)

---

## Motivasyon

Varlık **fiyatları** rastgele yürüyüşe (random walk) yakındır ve etkin piyasa hipotezi gereği yönleri güvenilir biçimde tahmin edilemez. Buna karşılık **oynaklık** tahmin edilebilirdir; çünkü kümelenir (clustering): sakin dönemleri sakin, çalkantılı dönemleri çalkantılı dönemler izler. Hedef değişkeni anlamlı kılan şey bu kalıcılıktır (persistence).

Hedef olarak fiyat/getiri yerine oynaklığın seçilmesi bilinçli bir metodolojik tercihtir: projeyi köklü bir istatistik literatürüne (ARCH/GARCH) dayandırır ve hem dürüst hem yorumlanabilir sonuçlar üretir. Açıklanabilirlik de burada kritiktir — finansta kara kutu bir model çoğu zaman yeterli değildir; modelin neden yüksek risk öngördüğünü anlamak güven, model riski yönetimi ve regülasyon açısından gereklidir.

## Problem Tanımı

Hedef, sonraki 5 işlem gününün logaritmik getirilerinin standart sapmasının yıllıklaştırılmış halidir:

```
target_vol(t) = std( ret[t+1 ... t+5] ) × √252
```

Pencere geleceğe kaydırılmıştır (`.shift(-5)`); böylece model, t anında bilinmeyen bir büyüklüğü tahmin eder. `√252` terimi yıllıklaştırma içindir ve yalnızca ölçek kolaylığı sağlar.

## Veri Seti

| Özellik | Değer |
|---|---|
| Kaynak | `yfinance` üzerinden Yahoo Finance (sembol `^GSPC`) |
| Varlık | S&P 500 endeksi |
| Dönem | 2010-01-01 – 2024-12-31 |
| Gözlem (öznitelik sonrası) | ≈ 3.700 işlem günü |
| Frekans | Günlük |
| Tahmin ufku | 5 işlem günü |

**Neden bu veri seti?** S&P 500, tek bir hisseye göre daha düşük gürültü içeren, literatürde yaygın bir benchmark'tır ve getiri, hacim, takvim gibi anlamlı dış değişkenler türetmeye uygundur; bu da XAI aşamasında modelin kararlarının yorumlanmasını sağlar.

## Öznitelik Mühendisliği

T�m öznitelikler yalnızca **geçmiş veriyi** kullanır; hedef ise geleceğe kaydırılır. Bu kesin ayrım, veri sızıntısını (data leakage) önler ve gerçekçi bir değerlendirme sağlar.

- **Gecikmeli oynaklık pencereleri** — `vol_5`, `vol_10`, `vol_22`, `vol_66` (getirilerin yıllıklaştırılmış hareketli std'si) — oynaklık kümelenmesini yakalar.
- **Getiri büyüklüğü** — `abs_ret_lag{1,2,3,5}` ve `ret_lag1` (dünkü getiri).
- **Hacim** — `vol_chg` (log hacim değişimi), `vol_ma22` (22 günlük hacim ortalaması).
- **Takvim** — `dow` (haftanın günü), `month` (ay).

> Teknik notlar: `log(0)` uyarısını önlemek için `log(volume + 1)` kullanılır; ortaya çıkabilecek `±inf` değerleri `NaN`'a çevrilip temizlenir.

## Yöntem

### Eğitim/test bölmesi ve doğrulama

Veri **kronolojik** olarak bölünür (%80 eğitim / %20 test, karıştırma yok). Hiperparametre seçimi, eğitim seti içinde **`TimeSeriesSplit`** (5 katman) ile yapılır.

> **Neden sıradan K-Fold değil?** Zaman serisinde rastgele K-Fold, modeli geleceğin verisiyle eğitip geçmişi tahmin ettirir — bu, sonuçları yapay biçimde iyileştiren bir sızıntıdır. `TimeSeriesSplit` her zaman geçmişle eğitip hemen sonraki dilimle doğrular ve zaman okunu korur.

### Modeller ve optimizasyon

| Model | Rolü | Optimizasyon |
|---|---|---|
| Naive (`vol_5`) | Referans (baseline) | yok — son 5 günü kopyalar |
| GARCH(1,1) | Klasik istatistik baseline | MLE (kayan/genişleyen pencere) |
| Ridge | Doğrusal temel model | Grid Search |
| Random Forest | Doğrusal-olmayan | Random Search |
| LightGBM | Gradient boosting | Bayesian (Optuna / TPE) |

Üç optimizasyon stratejisi bilinçli olarak farklı modellere eşlenmiştir: Grid Search küçük bir uzayı tüketici biçimde tarar, Random Search geniş bir uzayı verimli örnekler, Bayesian optimizasyon ise önceki denemelerden öğrenerek umutlu bölgelere yönelir. Tekrarlanabilirlik için her yerde `random_state=42` kullanılır.

## Sonuçlar

Test seti (son %20, yaklaşık 2022 sonu – 2024) üzerinde değerlendirilmiştir. En iyi değerler **kalın**.

| Model | RMSE | MAE | MAPE (%) | R² |
|---|---|---|---|---|
| Naive (`vol_5`) | 0.0825 | 0.0601 | 47.10 | 0.0124 |
| Ridge | 0.0726 | 0.0510 | 40.80 | 0.2360 |
| GARCH(1,1) | 0.0682 | 0.0515 | 45.03 | 0.3249 |
| **Random Forest** | **0.0669** | **0.0492** | **38.22** | **0.3508** |
| LightGBM | 0.0671 | 0.0494 | 38.38 | 0.3465 |

**Tablonun okunuşu:**

- **Naive baseline** R²'si neredeyse sıfırdır — son 5 günün oynaklığını kopyalamak neredeyse hiç açıklayıcı güç taşımaz, çünkü 5 günlük pencere kendisi gürültülü bir tahmincidir. Bu, gerçek modellerin yenmesi gereken eşiği belirler (RMSE 0.0825 / MAE 0.0601).
- **Ridge** bile baseline'ı her metrikte yener; bu, öznitelik mühendisliğinin işe yaradığının kanıtıdır.
- **Random Forest ve LightGBM istatistiksel olarak başa baştır** (RMSE 0.0669 vs 0.0671); fark üçüncü ondalıktadır.
- **GARCH(1,1)** yalnızca getiri kullanarak Ridge'i geçer ve ML modellerine çok yaklaşır — klasik yöntemlerin oynaklıkta ne kadar güçlü kaldığını gösterir.

**Model seçimi:** Başa baş duruma rağmen LightGBM seçilmiştir; çünkü aynı performansı çok daha sade bir modelle elde eder (`max_depth=2`, yalnızca 109 ağaç — Occam'ın usturası), Bayesian optimizasyonla ayarlanmıştır ve SHAP'ın exact `TreeExplainer`'ı ile temiz biçimde çalışır.

> **CV ile test üzerine bir not:** Random Forest'ın CV RMSE'si (0.0851), Ridge'inkinden (0.0771) *daha kötüydü*; ancak RF test setinde kazandı. Bu beklenen bir durumdur: erken `TimeSeriesSplit` katmanları çok az veriyle eğitildiğinden veri-açlığı çeken modelleri CV'de cezalandırır, oysa test modeli tüm eğitim verisini görür. CV ile tek-pencere test skorları farklı hikâyeler anlatabilir — dürüstçe raporlanmaya değer bir nokta.

## Açıklanabilir Yapay Zekâ (XAI)

En iyi model (LightGBM) üç bağımsız yöntemle analiz edilmiş; böylece bulgular çapraz doğrulanmıştır.

### 1. Permutation Importance (global)

Karıştırıldığında test RMSE'sini en çok artıran öznitelikler:

| Öznitelik | Önem | Std |
|---|---|---|
| `vol_22` | 0.00478 | 0.00047 |
| `vol_10` | 0.00434 | 0.00047 |
| `vol_5` | 0.00228 | 0.00035 |
| `ret_lag1` | 0.00188 | 0.00027 |
| `vol_ma22` | 0.00061 | 0.00015 |

Geçmiş oynaklık pencereleri baskındır — **oynaklık kümelenmesinin** doğrudan kanıtı. `vol_66` burada önemsiz görünür, çünkü bilgisi `vol_22`/`vol_10` ile örtüşür (permutation importance korelasyonlu öznitelikler arasında katkıyı paylaştırır) — bu da SHAP'ın neden ayrıca kullanıldığını açıklar.

### 2. SHAP (global + lokal)

- **Oynaklık kümelenmesi (yön ile) doğrulandı:** yüksek geçmiş oynaklık değerleri tahmini yukarı, düşük değerler aşağı iter.
- **Kaldıraç etkisi — en güçlü bulgu:** `ret_lag1` için negatif getiriler (düşüş günleri) tahmin edilen oynaklığı *yukarı* iter. Model bunu hiç öğretilmeden yalnızca veriden çıkarır. Permutation importance yalnızca "önemli" derken, SHAP etkinin *yönünü* ortaya koyar.
- **`vol_66` çözüldü:** SHAP, vol_66'nın gerçek bir etkisi olduğunu gösterir ve permutation'ın gizlediği bilgiyi geri kazandırır. İki yöntemin farklı sonuç vermesi hata değil, metodolojik zenginliktir.
- **Lokal açıklama (2022-04-28 kriz günü, gerçek oynaklık 0.462):** tahmin, baz değerden (0.134) 0.212'ye yükselir; başlıca etkenler `vol_10`, `vol_5`, `vol_22`'dir. Model krizi kümelenme yoluyla yakalar — ancak uç zirveyi hafife alır (regresyon modellerinin nadir olaylarda sık görülen bir özelliği).

### 3. LIME (lokal)

LIME, farklı bir mekanizmayla aynı kriz günü için aynı öznitelikleri (`vol_10`, `vol_5`, `vol_22`) öne çıkarır ve **bulguların yöntemden bağımsız (robust) olduğunu** doğrular.

## Kilit Bulgular

1. **Oynaklık tahmin edilebilir, fiyat değil** — problemi gerçekleşen oynaklık etrafında kurmak dürüst ve anlamlı sonuçlar üretir (en iyi R² ≈ 0.35, finansal oynaklık için makul bir değer).
2. **Klasik yöntemler rekabetçi kalır** — GARCH(1,1) ML modellerine neredeyse eşittir; ML'in asıl üstünlüğü ek sinyalleri (hacim, kaldıraç) entegre edebilmesi *ve* yorumlanabilir olmasıdır.
3. **Model iki olguyu yeniden keşfeder** — oynaklık kümelenmesi ve kaldıraç etkisi — doğrudan veriden, üç bağımsız XAI yöntemiyle doğrulanmış olarak.
4. **Metodoloji önemlidir** — kronolojik bölme, `TimeSeriesSplit`, sızıntıya kapalı öznitelikler ve dürüst baseline'lar sonuçları güvenilir kılar.

## Proje Yapısı

```
.
├── notebook.ipynb              # Uçtan uca analiz (veri → modeller → XAI)
├── README.md                   # Bu dosya
├── report/                     # Proje raporu (PDF/DOCX)
├── figures/                    # Dışa aktarılan grafikler (önem, SHAP, LIME)
└── requirements.txt            # Bağımlılıklar
```

> Yukarıdaki dosya/klasör adlarını kendi deponun gerçek yapısına göre düzenle.

## Kurulum

```bash
git clone https://github.com/<kullanici-adin>/<depo-adi>.git
cd <depo-adi>
pip install -r requirements.txt
```

Temel bağımlılıklar:

```
yfinance
pandas
numpy
scikit-learn
lightgbm
xgboost
optuna
shap
lime
arch
matplotlib
```

## Kullanım

Notebook'u açıp hücreleri baştan sona çalıştır:

```bash
jupyter notebook notebook.ipynb
```

Notebook veriyi indirir, öznitelikleri üretir, tüm modelleri eğitip ayarlar ve karşılaştırma tablosu ile XAI grafiklerini oluşturur. `yfinance` indirmesi için internet bağlantısı gereklidir.

## Sınırlılıklar ve Gelecek Çalışma

- Tek bir test penceresi kullanılmıştır; **walk-forward doğrulama** kararlılığı daha iyi ölçer.
- Yalnızca tek bir varlık (S&P 500) incelenmiştir; **çoklu varlık** genellenebilirliği sınar.
- Tek bir ufuk (5 gün) kullanılmıştır; daha kapsamlı bir **forecast-horizon analizi** eklenebilir.
- ML modelleri sabit eğitim seti, GARCH ise kayan pencere kullanır — ML modellerine de walk-forward uygulamak karşılaştırmayı tam simetrik yapar.
- Asimetrik GARCH varyantları (**EGARCH, GJR-GARCH**) kaldıraç etkisini doğrudan modeller; baseline olarak eklenebilir.

## Kaynaklar

- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS — SHAP.
- Ribeiro, Singh & Guestrin (2016), *"Why Should I Trust You?"*, KDD — LIME.
- Akiba ve diğerleri (2019), *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD.
- Engle (1982), *Autoregressive Conditional Heteroscedasticity*, Econometrica — ARCH.
- Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*, Journal of Econometrics — GARCH.
- Mandelbrot (1963), *The Variation of Certain Speculative Prices* — oynaklık kümelenmesi.
- Black (1976) — kaldıraç etkisi.
- scikit-learn (Pedregosa ve diğerleri, 2011), LightGBM (Ke ve diğerleri, 2017), `arch` (Sheppard), `yfinance`.

> Her kaynağın tam künyesini son kullanımdan önce doğrula.

---

*Bu proje, optimizasyon tabanlı model seçimi ve açıklanabilir yapay zekâ üzerine bir ders/portfolyo çalışması olarak geliştirilmiştir.*
