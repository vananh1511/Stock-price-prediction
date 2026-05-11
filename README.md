**Deep Learning Final Project**

**Stock Price Prediction & Signal Classification**

*Four-Task Pipeline \| Conv1D · GRU · Rule-Based Signals · Portfolio
Optimisation*

  -----------------------------------------------------------------------
  Student ID         210186
  ------------------ ----------------------------------------------------
  Tasks              Task 1 (NVDA) \| Task 2 (VNM) \| Task 3+4 (VNG DL +
                     Portfolio)

  Language           Python 3.10

  Framework          TensorFlow 2.20 · Keras Tuner 1.4 · scikit-learn ·
                     PyPortfolioOpt
  -----------------------------------------------------------------------

![](media/image1.png){width="6.5in" height="4.260416666666667in"}

# 1. Overview

This project implements deep learning pipeline for stock price
prediction and signal classification. Tasks 1 and 2 focus on price
prediction using Conv1D-GRU architectures. Task 3 buy/sell signal
identification was first examined with simple rule-based approach. Task
3+4 (the combined notebook) extends the rule-based signals with a
trained meta-model and optimises a portfolio using maximum Sharpe ratio.

  -----------------------------------------------------------------------------------
  **Task**   **Notebook**                         **Stock**     **Objective**
  ---------- ------------------------------------ ------------- ---------------------
  Task 1     210186\_ Task1.ipynb                 NVDA (NASDAQ) Price regression ---
                                                                next-day, k-day ahead
                                                                and several days
                                                                ahead

  Task 2     210186_Task2.ipynb                   VNM (VNINDEX) Price regression ---
                                                                next-day, k-day ahead
                                                                and several days
                                                                ahead

  Task 3     210186_Rule_based_Task_3_VNG.ipynb   VNG (VNINDEX) Rule-based
                                                                accumulation /
                                                                distribution signals

  Task 3+4   210186\_ Task3_4_Portfolio.ipynb     VNG + 11      Meta model combined
                                                  stocks        DL and rule-based
                                                                system for predicting
                                                                buying/selling
                                                                signals + portfolio
                                                                optimisation
  -----------------------------------------------------------------------------------

# 2. Project Files

## 2.1 Notebooks

  --------------------------------------------------------------------------------
  **File**                             **Description**
  ------------------------------------ -------------------------------------------
  210186\_ Task1.ipynb                 Task 1 --- NVDA price prediction
                                       (next-day + k-day + 3-day walk-forward)

  210186\_ Task2.ipynb                 Task 2 --- VNM price prediction (same
                                       architecture, Vietnamese stock)

  210186_Rule_based_Task_3_VNG.ipynb   Task 3 --- Rule-based HH/AD/EMA signal
                                       backtesting on VNG

  210186\_ Task3_4_Portfolio.ipynb     Task 3+4 --- Meta-model predicting
                                       buying/selling signals + portfolio
                                       optimisation
  --------------------------------------------------------------------------------

## 2.2 Data Files

  ------------------------------------------------------------------------
  **File**                  **Contents**
  ------------------------- ----------------------------------------------
  NVDA.csv                  NVDA daily (Date, Open, High, Low, Close,
                            Volume)

  MU.csv                    Micron daily (Date, Open, High, Low, Close,
                            Volume)

  VNM-VNINDEX-History.csv   VNM daily (TradingDate, Open, High, Low,
                            Close, Volume)

  VNG-VNINDEX-History.csv   VNG daily OHLCV --- used in Task 3
                            (rule-based) and Task 3+4 (DL training stock)

  Medical sector/ Banking   Task 4 universe --- 11 Vietnamese stocks (same
  sector folder             CSV format as VNG)
  ------------------------------------------------------------------------

# 3. Task 1 --- NVDA Price Prediction

## 3.1 Objective

Predict NVDA open price using a Conv1D-based regression model. Three
progressively advanced sub-tasks are implemented within the same
notebook:

-   v1: Predict next-day open price using Open + Close features (2
    features)

-   v2: Predict next-day open price using Open, Close, Low, High (4
    features) with volatile data augmentation and walk-forward
    retraining

-   v3: Predict 3 days ahead (multi-output regression) using 4 features
    with walk-forward prediction

-   Apply v3 model to Micron stock price data to examine generalization

## 3.2 Architecture

  ------------------------------------------------------------------------
  **Version**        **Architecture**                        **Loss**
  ------------------ --------------------------------------- -------------
  v1 (next-day, 2    Conv1D(64) → Pool → Conv1D(128) → Pool  MSE
  feat.)             → Conv1D(64) → Pool → Flatten →         
                     Dense(100) → Dense(1)                   

  v2 (next-day, 4    Conv1D(64) → Pool → GRU(64, seq) →      Asymmetric
  feat.)             GRU(32) → Dense(32) → Dense(1)          MSE (alpha=2)

  v3 (3-day ahead)   Conv1D(64) → Pool → GRU(64, seq) →      MSE per day
                     GRU(32) → Dense(32) → Dense(3)          
  ------------------------------------------------------------------------

## 3.3 Key Techniques

-   Per-sample min-max normalisation --- each 30-bar window scaled to
    \[0, 1\] independently

-   Volatile data augmentation --- synthetic windows with scaled
    volatility (scale 2-10x, 5 copies) to improve robustness

-   Asymmetric MSE --- penalises underprediction (alpha=2) more than
    overprediction to reduce downside risk

-   Walk-forward prediction with periodic retraining every 30 test
    samples (5 fine-tune epochs) to adapt to regime changes

-   Grid search over learning rates \[0.001, 0.0005\] and batch sizes
    \[32, 64\]

## 3.4 Constants

  --------------------------------------------------------------------------
  **Constant**       **Value**   **Description**
  ------------------ ----------- -------------------------------------------
  window_size        30          Number of bars per input window

  n_features v1      2           Open, Close

  n_features v2/v3   4           Open, Close, Low, High

  n_ahead v3         3           Days ahead to predict simultaneously

  retrain_every      30          Walk-forward retraining interval (test
                                 samples)

  finetune_epochs    5           Fine-tuning epochs per walk-forward step
  --------------------------------------------------------------------------

# 4. Task 2 --- VNM Price Prediction

## 4.1 Objective

Replicate the Task 1 pipeline on VNM (Vietnam Dairy Products
Corporation, VNINDEX). The same three sub-tasks (v1, v2, v3) are
implemented with identical architecture, demonstrating that the approach
generalises from a US tech stock to a Vietnamese market stock.

## 4.2 Differences from Task 1

  ------------------------------------------------------------------------
  **Aspect**         **Task 1 (NVDA)**          **Task 2 (VNM)**
  ------------------ -------------------------- --------------------------
  Data source        NVDA.csv (Nasdfaq, Date    VNM-VNINDEX-History.csv
                     column)                    (TradingDate column)

  Price scale        USD (tens to hundreds of   VND (thousands, different
                     dollars)                   magnitude)

  Market             NASDAQ --- highly liquid,  VNINDEX --- lower
                     continuous trading         liquidity, local holidays

  Architecture       Identical                  Identical

  Augmentation       scale_range=(2, 10)        scale_range=(2, 5) - lower
                                                due to different
                                                volatility profile
  ------------------------------------------------------------------------

# 5. Task 3 --- Rule-Based Signal Analysis (VNG)

## 5.1 Objective

Identify accumulation (buy) and distribution (sell) zones on VNG using
three technical indicators combined as a rule-based filter. Backtest hit
rates and profit factors across four forward horizons.

## 5.2 Signals

  -------------------------------------------------------------------------
  **Indicator**   **Condition**                      **Interpretation**
  --------------- ---------------------------------- ----------------------
  HH (Higher      1 if current High \> 14-bar        Bullish --- buyers
  High)           rolling max of prior Highs         breaking resistance

  LL (Lower Low)  1 if current Low \< 14-bar rolling Bearish --- sellers
                  min of prior Lows                  breaking support

  AD_slope        20-bar diff of rolling-normalised  Positive = buying
                  Accumulation/Distribution line     pressure; negative =
                                                     selling

  EMA9_ratio      Close / EMA(9)                     \>1 = price above
                                                     trend (bullish); \<1 =
                                                     below
  -------------------------------------------------------------------------

## 5.3 Zone Definition

+-----------------------------------------------------------------------+
| **Accumulation zone:**                                                |
|                                                                       |
| HH == 1 AND AD_slope \> 0 AND EMA9_ratio \> 1                         |
|                                                                       |
| **Distribution zone:**                                                |
|                                                                       |
| LL == 1 AND AD_slope \< 0 AND EMA9_ratio \< 1 AND vol_ratio \> 1.1    |
|                                                                       |
| All conditions must fire simultaneously for a zone signal.            |
+=======================================================================+
+-----------------------------------------------------------------------+

## 5.4 Backtesting

Hit rates and average returns are computed over four forward horizons:
5, 10, 20, and 30 bars. Horizon numbers were chosen empirically based on
higher hit rates. Metrics include hit rate (directional accuracy),
average return, and profit factor.

  -----------------------------------------------------------------------
  **Metric**         **Definition**
  ------------------ ----------------------------------------------------
  Hit rate           Fraction of signal bars where price moved in the
                     predicted direction after N bars

  Avg return         Mean forward return on all signal bars (%)

  Profit factor      Gross winning returns / gross losing returns
                     (absolute)
  -----------------------------------------------------------------------

# 6. Task 3+4 --- DL Signal Classifier + Portfolio Optimisation

## 6.1 Objective

Train a deep learning binary classifier on VNG to identify accumulation
vs distribution zones, then transfer the trained model to a universe of
11 Vietnamese stocks and construct an optimised portfolio.

## 6.2 Notebook Structure

  --------------------------------------------------------------------------
  **§**   **Description**
  ------- ------------------------------------------------------------------
  0       Imports, shared constants, stock universe definition

  1       Shared utility functions (load_csv, engineer_features,
          build_windows, minmax_normalize, compute_rule_score)

  2       Load & prepare training data (VNG-VNINDEX-History.csv)

  3       Sliding windows + normalisation

  4       Chronological train / val / test split (split before
          normalisation)

  5       Model architecture (Conv1D-GRU + focal loss)

  6       Bayesian hyperparameter search (Keras Tuner, 15 trials)

  7       Final model training

  8       Training curves (loss, accuracy, AUC)

  9       Meta-learner: rule score + GRU probability blend (Logistic
          Regression)

  10      Test set evaluation: classification report, AUC-ROC, head-to-head
          comparison

  11      Confidence analysis

  12      Overlay predictions on price chart

  13      Save artefacts (task3_gru_model.keras, task3_meta_model.pkl)

  14      Load Task 4 stock universe

  15      EDA: return distributions, correlation heatmap (date-aligned),
          rolling volatility, ADF test

  16      Cross-stock inference on 20-bar horizon

  17      Results table: avg return, hit rate, profit factor

  18      Task 4 visualisations

  19      Portfolio selection (hit rate \> 52%)

  20      Price matrix --- date-aligned via TradingDate inner join

  21      Expected returns (80% historical + 20% model signal) and
          Ledoit-Wolf covariance

  22      Maximum Sharpe ratio optimisation (PyPortfolioOpt)

  23      Equal-weight benchmark

  24      Discrete allocation (share counts from PORTFOLIO_BUDGET)

  25      Final portfolio summary
  --------------------------------------------------------------------------

## 6.3 Model Architecture (Task 3 DL)

  -----------------------------------------------------------------------
  **Layer**          **Detail**
  ------------------ ----------------------------------------------------
  Input              (20 bars × 9 features) sliding window

  Conv1D             64 filters, kernel=3, padding=same --- local pattern
                     extraction

  MaxPooling1D       pool=2

  Dropout            tuned 0.1--0.2

  GRU layer 1        32 or 64 units, return_sequences=True

  Dropout            tuned

  GRU layer 2        16, 32, or 64 units

  Dropout            tuned

  BatchNorm          stabilises training

  Dense(32)          ReLU

  Dense(1)           Sigmoid --- output: P(Accumulation)

  Loss               Focal loss (gamma=2.0, alpha=0.4) --- handles class
                     imbalance
  -----------------------------------------------------------------------

## 6.4 Features (9 total --- same in rule-based and DL)

  -----------------------------------------------------------------------
  **Feature**           **Description**
  --------------------- -------------------------------------------------
  Open, High, Low,      Raw OHLC prices --- per-sample min-max normalised
  Close                 

  Volume                Raw trading volume

  HH                    1 if High \> prior 14-bar rolling max; 0
                        otherwise

  LL                    1 if Low \< prior 14-bar rolling min; 0 otherwise

  AD_slope              20-bar diff of 30-bar normalised A/D line

  EMA9_ratio            Close / EMA(9)
  -----------------------------------------------------------------------

## 6.5 Task 4 Stock Universe

  ---------------------------------------------------------------------------
  **Ticker**   **CSV File**                       **Sector**
  ------------ ---------------------------------- ---------------------------
  DHG          DHG-VNINDEX-History.csv            Healthcare

  DHN          DHN-UpcomIndex-History.csv         Healthcare

  DBT          DBT-VNINDEX-History.csv            Healthcare

  DCL          DCL-VNINDEX-History.csv            Healthcare

  DHD          DHD-UpcomIndex-History.csv         Healthcare

  DPH          DPH-UpcomIndex-History.csv         Healthcare

  HDP          HDP-UpcomIndex-History.csv         Healthcare

  BID          BID-VNINDEX-History.csv            Banking

  VCB          VCB-VNINDEX-History.csv            Banking

  VIB          VIB-VNINDEX-History.csv            Banking

  TCB          TCB-VNINDEX-History.csv            Banking
  ---------------------------------------------------------------------------

## 6.6 Portfolio Optimisation

  -----------------------------------------------------------------------
  **Parameter**         **Value / Method**
  --------------------- -------------------------------------------------
  Selection threshold   Hit rate \> 52% on 5-bar horizon

  Expected returns      80% mean historical return + 20% model signal
                        return (annualised)

  Covariance            Ledoit-Wolf shrinkage on shared-date price matrix

  Optimisation          EfficientFrontier.max_sharpe() --- long-only (0-1
                        bounds)

  Risk-free rate        4.3% (Vietnamese 10-year government bond)

  Date alignment        TradingDate inner join
  -----------------------------------------------------------------------

# 7. Shared Pipeline Design

## 7.1 Per-Sample Min-Max Normalisation

All four tasks use the same normalisation approach: for each input
window, the min and max are computed over that window alone and all
price features are scaled to \[0, 1\]. This ensures no future data
contaminates normalisation, and the model is invariant to absolute price
levels.

## 7.2 Chronological Split

All train/val/test splits use shuffle=False to preserve time order. The
split is performed before normalisation to prevent any cross-set
leakage.

  -----------------------------------------------------------------------
  Tasks 1 & 2        80% train / 20% test, then 80% of train / 20% val
  ------------------ ----------------------------------------------------
  Task 3+4           50% train / 30% val / 20% test

  -----------------------------------------------------------------------

## 7.3 Walk-Forward Prediction (Tasks 1 & 2)

After training, predictions on the test set are made in chunks of 30
samples. After each chunk, the model is fine-tuned for 5 epochs on the
augmented training data so it adapts to evolving market regimes. This is
the production-simulation approach.

## 7.4 Volatile Data Augmentation (Tasks 1 & 2)

Synthetic training windows are created by scaling volatility: for each
window, deviations from the mean are amplified by a random factor. This
teaches the model to handle price regimes it may not have seen during
the training period.

+-----------------------------------------------------------------------+
| X_volatile = mean + (X_norm - mean) \* scale                          |
|                                                                       |
| scale \~ Uniform(2, 10) for NVDA                                      |
|                                                                       |
| scale \~ Uniform(2, 5) for VNM                                        |
|                                                                       |
| n_copies = 5 (6x original dataset size after augmentation)            |
+=======================================================================+
+-----------------------------------------------------------------------+

# 8. Requirements

## 8.1 Python Packages

pip install tensorflow keras-tuner scikit-learn pandas numpy matplotlib
seaborn

pip install statsmodels scipy PyPortfolioOpt

## 8.2 Data Files Required

  --------------------------------------------------------------------------------------
  **File**                                  **Used   **Format**
                                            in**     
  ----------------------------------------- -------- -----------------------------------
  NVDA.csv                                  Task 1   Date, Open, High, Low, Close,
                                                     Volume

  VNM-VNINDEX-History.csv                   Task 2   TradingDate, Open, High, Low,
                                                     Close, Volume

  VNG-VNINDEX-History.csv                   Task 3 & TradingDate, Open, High, Low,
                                            3+4      Close, Volume

  DHG/DHN/DBT/DCL/DHD/DPH/HDP-History.csv   Task 4   Same format as VNG --- healthcare
                                                     stocks

  BID/VCB/VIB/TCB-History.csv               Task 4   Same format as VNG --- banking
                                                     stocks
  --------------------------------------------------------------------------------------

# 9. How to Run

## 9.1 Tasks 1 & 2

1.  Place NVDA.csv (Task 1) or VNM-VNINDEX-History.csv (Task 2) in the
    notebook directory.

2.  Open the respective notebook in Jupyter or Google Colab.

3.  Run all cells top to bottom. Section 1.1 trains the next-day model;
    1.2 extends to k-day and walk-forward.

4.  Estimated runtime: \~10 min on CPU, \~3 min on Colab GPU.

## 9.2 Task 3 (Rule-Based)

5.  Place VNG-VNINDEX-History.csv in the notebook directory.

6.  Open 210186_Rule_based_Task_3_VNG.ipynb.

7.  Run all cells. No GPU required --- pure pandas/numpy computation.

## 9.3 Task 3+4 (DL + Portfolio)

8.  Place VNG and all 11 universe CSV files in the notebook directory.

9.  Open 210186\_ Task3_4_Portfolio.ipynb.

10. Run all cells. Sections 0-13 = Task 3 DL training. Sections 14-25 =
    Task 4 evaluation and portfolio.

11. Bayesian search (Section 6) takes \~10-20 min on CPU.

12. After running Section 13, two artefact files are saved:
    task3_gru_model.keras and task3_meta_model.pkl.

# 10. Key Constants Reference

  ------------------------------------------------------------------------
  **Constant**       **Value**          **Description**
  ------------------ ------------------ ----------------------------------
  window_size        30 (T1/T2) / 20    Input window length in bars
                     (T3+4)             

  n_features         2 or 4 (T1/T2) / 9 Number of input features per bar
                     (T3+4)             

  HORIZON_T3         5                  Label look-ahead for Task 3+4
                                        training

  HORIZON_T4         5                  Signal evaluation horizon for Task
                                        4

  THRESHOLD          0.015              ±1.5% forward-return threshold for
                                        label assignment

  HIT_RATE_MIN       0.52               Minimum hit rate for portfolio
                                        inclusion

  RISK_FREE_RATE     0.043              4.3% annualised risk-free rate

  SIGNAL_BLEND       0.2                Weight of model signal in expected
                                        return estimate

  retrain_every      30                 Walk-forward retraining interval
                                        (T1/T2)

  n_copies           5                  Augmentation copies per training
                                        window (T1/T2)

  SEED               42                 Global random seed
  ------------------------------------------------------------------------

# 11. Known Limitations

-   Tasks 1 & 2: walk-forward fine-tuning uses the full augmented
    training set, which grows large for long histories. Memory usage
    scales linearly with dataset size.

-   Task 3 (rule-based): the three-condition AND filter is strict ---
    signal frequency drops significantly in trending markets where one
    condition may not fire.

-   Task 4: SIGNAL_BLEND=0.2 still incorporates in-sample signal quality
    into expected returns. For live deployment, use only out-of-sample
    signal returns.

-   Date alignment in Task 4 uses an inner join on TradingDate. Stocks
    with very different listing dates or exchange holidays will produce
    shorter shared windows, potentially reducing covariance estimate
    quality.
