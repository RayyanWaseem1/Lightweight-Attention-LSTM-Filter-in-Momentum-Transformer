# Introduction

## Research Objective
This project aims to develop a production-ready momentum trading strategy that addresses these challenges through several key objectives.

### Generalization Across Assets:
I developed a stock-agnostic feature engineering pipeline that transforms raw OHCLV (Open, High, Low, Close, Volume) data into normalized, dimensionless features that maintain consistent semantics across different securities. This enables a single model to trade large-cap technology stocks, emerging market ETFs, and commodity futures without retraining. 

### Adaptive Architecture Selection:
Rather than committing to a single neural architecture, I implemented a regime-aware ensemble that dynamically weights predictions from multiple models based on the current market state. The system learns to recognize when simpler vanilla transformers suffice versus when more complex attention-enhanced architectuers are needed, optimizing the bias-variance tradeoff in real time 

### Realistic Performance Assessment:
I constructed a comprehensive backtesting framework that incorporates realistic trading constraints including transaction costs (modeled at 5, 10, and 20 basis points), slippage, position limits, and weekly rebalancing delays. Walk-forward validation ensures that all results reflect genuinely out-of-sample performance without any lookahead bias.

### Interpretability and Robustness:
Beyond achieving strong annualized returns, I prioritized understanding why the model works through feature importance analysis and regime-specific performance decomposition. This interpretability is crucial for risk management and regulatory compliance in production deployments.

## Key Contributions
The main contributions of this project include: 

- Stock-Agnostic Feature Framework: A comprehensive feature engineering methodology that enables model generalization across 52+ diverse assets spanning multiple sectors, geographies, and market capitalizations. My fetaures are carefully designed to be dimensionless, normalized, and free from lookahead bias.
- Regime-Aware Ensemble Architecture: A novel ensemble approach that extracts 21 - dimensional regime features from input seequences and uses a learned weight network to dynamically allocate between vanilla and attention-enhanced transformers. Empirical results show that the ensemble automatically learns to use vanilla models 85.7% of time time (in stable markets) and switches to attention models 14.3% of the time (during regime transitions).
- Production-Grade Backtesting: A rigorous evaluation framework incorporating walk-forward validation, multiple transaction cost scenarios, portfolio construction rules, and comprehensive performance metrics. My results survive pessimistic assumptions (20 bps costs, weekly rebalancing) while still being able to generate substantial alpha.
- Comprehensive Ablation Analysis: Systematic evaluation of model components reveal that regime-aware switching is essential. Neither vanilla nor attention models alone achieve positive risk-adjusted returns, validating my architectural choices.

# Architecture and Design

## Overview of Model Architecture
The system implements three distinct transformer-based architectures, each designed to capture different aspects of momentum dynamics. Understanding the motivation and mechanics of each architecture is crucial for appreciating why the ensemble approach proves superior. 

### Vanilla Transformer (MomentumTransformerSimple)
The vanilla architecture represents my baseline approach, combining local feature extraction via LSTM layers with global dependency modeling through transformer encoders. The design follows the principle of hierarchical processing: lower layers capture short-term patterns (hours to days) while upper layers integrate long-term trends (weeks to months). 

The vanilla architecture embodies simplicity and efficiency. The LSTM encoder acts as a preprocessing filter, condensing raw features into a compact representation that emphasizes recent patterns most relevant for momentum prediction. By restricting the LSTM to a 63-timestep window, I was able to prevent the model from overfitting to long-term patterns that may not persist. The transformer layers then operate on this filtered representation, using attention to identify which historical periods are most predictive of future returns. 

This two-stage design is computationally efficient (1.45 ms inference time) and relatively resistant to overfitting due to its limited capacity (~50k parameters). However, the vanilla architecture has a critical limitation: its fixed processing pipeline cannot adapt to regime changes. During stable trending markets, the simple feature extraction suffices. But during high-volatility periods or regime transitions, more sophisticated feature processing is needed -- motivating my attention-enhanced variant. 

### Attention Enhanced Transformer (MomentumTransformerDualPath)
The attention-enhanced architecture addresses the vanilla model's inflexibility by introducing a dual-path LSTM encoder with explicit attention mechanisms and gating. This design is motivated by the observation that different time periods require different feature extraction strategies: during stable markets, broad averaging works well, but during volatile periods, the model must focus on specific critical events. 

The core innovation lies in the LSTM encoder, which processes inputs through two parallel pathways. 
- Main Path: A standard LSTM that processes the sequence sequentially, maintaining a running summary of historical patterns. this path captures general trends and provides a stable baseline representation. 
- Attention Path: A lightweight attention mechanism that identifies which specific timesteps are most relevant.
- Gated Combinations: Rather than simply averaging the two paths, we learn when to rely on each. This gating mechanism allows the model to dynamically adjust its processing strategy. During normal market conditions, the gate learns to favor the main path, effecitvely reducing to the vanilla architecture. But when the attention mechanism detects anomalous patterns (e.g., sudden volatility spikes, volume surges), the gate opens to incoporate this information.

Mathematical Formulation: 
The attention mechanism computes scores for each hidden state in the sequence:
<img width="702" height="164" alt="Screenshot 2026-01-27 at 6 59 47 PM" src="https://github.com/user-attachments/assets/816e394d-5d6a-4470-856a-f7f3fa7845b5" />

The context vector aggregates information across timesteps weighted by attention:
<img width="700" height="46" alt="Screenshot 2026-01-27 at 7 00 43 PM" src="https://github.com/user-attachments/assets/99161dd0-f2a3-4061-a3d1-335f4dddbc1e" />

Finally, the gated combination produces the enhanced representation:
<img width="695" height="163" alt="Screenshot 2026-01-27 at 7 01 18 PM" src="https://github.com/user-attachments/assets/80bee851-2c44-4fe8-ad79-4a96e0113b03" />

However, the attention-enhanced architecture has approximately 75k parameters (50% more than vanilla) and slower inference (1.52 ms vs 1.45 ms). Interestingly, when used independently, it achieves negative returns (-21% over the test period), suggesting it overfits to volatile training periods and fails to generalize. However, this architecture becomes highly valuable when combined with the vanilla model through regime-aware weighting. 

### Regime-Aware Ensemble (EnsembleMomentumTransformer)
The ensemble architecture represents my key innovation: rather than choosing between vanilla and attention-enhanced models, we can learn when to use each. This is motivated by the fundamental regime-switching nature of financial markets - different architectures have complementary strengths that can be exploited through intelligent combinations. 

Ensemble Architecture:
<img width="695" height="506" alt="Screenshot 2026-01-27 at 7 04 04 PM" src="https://github.com/user-attachments/assets/a7e1b0f4-127a-4345-9878-2a88f32a390c" />

The regime extractor computes 21 features that characterize the current market state. These features are carefully designed to capture different aspects of market dynamics. 

- Stock-Level Features (13 dimensions):
  - Volatility Regime Indicators: I computed short-term (21-day) and long-term (63-day) volatility using rolling standard deviations of returns. The ratio sigma_short / sigma_long indicates whether volatility isi expanding or contracting. I also created one-hot encodings for three volatility regimes: high (sigma > 3%), mediium (1% <= sigma <= 3%), and low (sigma < 1%). These thresholds are based on typical equity volatility levels and help the model recognize when markets are stressed.
  - Trend Strength: Measured via lag-1 autocorrelation of returns. High positive autocorrelation (>0.3) indicates trending behavior where momentum strategies excel. Negative autocorrelation suggests mean reversion where momentum fails. This single feature effectively captures the fundamental regime distinction most relevant to momentum trading.
  - Statistical Moments: I computed mean returns, absolute mean returns (to capture volatility regardless of direction), return range (max-min over window), and skewness. Skewness is particularly important because it indicates asymmetry in the return distribution -- negative skewness often precedes market crashes and should trigger defensive positioning.
  - Momentum Indicators: Short-term (21-day) and long-term (63-day) momentum is calculated as cumulative returns over the window. I also include raw values from technical indicators (RSI, realized volatility, momentum features) to give the model access to traditional signals.
 
- Cross-Sectional Features (5 dimensions):
  - Relative Return: The difference between the stock's 21-day return and the market's (SPY) 21-day return. Stocks with high relative return are exhibiting idiosyncratic strength that may persist.
  - Relative Volatility Ratio: Stock volatility divided by market volatility. A ratio of 2.5+ indicates the stock is much more volatile than the market (common during crises or stock-specific events). This feature helps the model recognize when stock-specific risk is elevated.
  - Beta: The correlation between stock and market returns over a 63-day window. High beta stocks amplify market movements; low beta provides diversification. During market downturns, the model should favor low-beta positions.
  - Relative Strength: Cumulative outperformance (stock return - market return) over 21 days. This captures sustained leadership/laggard behavior.
  - Market Volatility Level: Absolute market volatility (not relative to the stock). This provides context about overall market stress -- even if a stock has low relative volatility, high market volatility should trigger more conservative positioning.

- Temporal Features (3 dimensions):
  - Volatility Persistence: I computed volatility over rolling 21-day windows and measure the standard deviation of these volatility estimates. Low persistence means volatility recently spiked; high persistence means volatility has been stable. Recent spikes often precede regime changes.
  - Volatility Trend: Teh change in volatility over time (recent vol minus historical vol). Positive values indicate rising volatility, which typically precedes market stress.
  - Regime Shift Indicator: The absolute difference between recent mean return and historical mean return. Large values indicate the statistical properties of returns have changed, suggesting a regime transition.

The weight network is a simple feedforward neural network that learns to map regime features to model weights. The constraints to [0.2, 0.8] is crucial: it ensures that both models always contribute at least 20% to the final prediction. This prevents the network from degenerating to a single model and maintains the ensemble diversity. During training, the network learns to increase the attention weight when:
  - volatility is spiking (high vol_ratio, high vol_trend)
  - regime shifts are detected (high regime_shift indicator)
  - autocorrelation is breaking down (trend_strength near zero)
  - market stress is elevated (high market volatility)
Conversely, it learns to favor the vanilla model during:
  - stable trending markets (high positive autocorrelation)
  - low volatility environments
  - consistent momentum patterns

The ensemble is trained end-to-end using the Sharpe ratio loss. Critically, the weight network's gradients come from the final portfolio loss, not from auxiliary objectives. This means that the network learns to weight models based on their contribution to risk-adjusted returns, not on abstract notions of "regime classification." If using the attention model during a particular market state improves Sharpe, the weight network learns to increase attention weights in similar future states. 

Analysis of the trained ensemble on the test set reveals fascinating behavior:
  - Mean attention weight: 0.411 (41%). This means that on average, in the ensemble, attention had ~40% of a contribution, compared to ~60% from just the vanilla. Meaning, they worked in tandem together on about a 60/40 split.
  - Attention model usage (weight > 0.5): 14.3% of periods. This means that there were only ~14% of periods where the model primarily used attention. That threshold was set to 0.5.
  - Vanilla model ussage (weight > 0.5): 85.7% of periods. Again this means that ~85% of the time, vanilla had more of a pulling factor as compared to the attention model.

The ensemble primarily relies on the vanilla model but makes strategic use of the attention model during specific periods. Correlation analysis revelas that attention weights spike during:
  - Q1 2023: Regional banking crisis (SVB collapse)
  - August 2023: Credit rating downgrade volatility
  - October 2023: Middle East tensions
  - March 2024: Inflation concerns

During these periods, the attention model's ability to focus on specific critical events provides value. But in normal markets, the vanilla model's simplicity prevents overfitting. 

## Mathematical Foundations
### Attention Mechanisms
### Multi-Head Self-Attention in Transformers:
The transformer's core innovation is the attention mechanism, which allows the model to weight the importance of different positions in the input sequence when computing representations. Unlike RNNs that process sequences left-to-right, attention provides global connectivity: each position can directly attend to all others. 

The basic attention operations computes a weighted sum of value vectors, where weights are determined by compatibility between query and key vectors:
<img width="698" height="208" alt="Screenshot 2026-01-27 at 7 27 35 PM" src="https://github.com/user-attachments/assets/9d2adfd7-426e-4ef1-8073-80c6d6122902" />

The scaling factor sqrt(d_k) prevents the dot products from growing too large, which would push the softmax into regions with vanishingly small gradients. 

Multi-head attention applies this operation in parallel with different learned projections. 
<img width="692" height="162" alt="Screenshot 2026-01-27 at 7 28 40 PM" src="https://github.com/user-attachments/assets/dec4645d-0622-4113-8c4f-a6439570d476" />

Using multiple heads allows the model to attend to different types of patterns simultaneously. For example, one head might focus on recent returns, another on volatility spikes, and a third on volumen anomalies. The concatenated outputs are then projected back to the model dimension.

### Why Attention for Financial Time Series?
Attention is particularly valuable for financial data because important patterns often occur at irregular intervals. Traditional momentum might rely on fixed lookback windows (e.g., 20-day moving average), but attention learns which historical points are relevant. For instance, the attention mechanism might learn to:
  - focus on the last major earnings announcement regardless of when it occurred
  - identify previous periods with similar volatility patterns
  - detect regime changes by attending to historical transitions
  - ignore periods of low information content (e.g., low-volume trading days)

### Dual-Path LSTM Attention:
My attention-enhanced architecture uses a different attention variant designed specifically for sequence encoding. Unlike transformer self-attention which operates on all positions, LSTM attention computes a context vector by weighting hidden states:
<img width="700" height="336" alt="Screenshot 2026-01-27 at 7 32 06 PM" src="https://github.com/user-attachments/assets/c64d96bd-83f0-4e8d-8275-6d8c69b1df5e" />

This context vector aggregates information from all timesteps, with weights determined by the learned compatibility. The attention mechanism essentially learns a soft, differentiable version of "which historical point is most similar to the current state?"

The gating mechanism then decides how much to rely on this context:
<img width="700" height="223" alt="Screenshot 2026-01-27 at 7 33 18 PM" src="https://github.com/user-attachments/assets/d3ddfcc0-ce54-4655-af77-0093e600fa94" />

This formulation is inspired by gated recurrent units (GRUs) and allows the model to learn complex, conditional processing strategies. The gate can learn patterns like "when volatility is high, use attention to find similar historical high-vol periods; otherwise, use standard LSTM processing."

### Loss Functions and Optimizations
My primary training objective is maximizing the Sharpe ratio, a standard measure of risk-adjusted returns in finance. Unlike classification or regression problems with ground-truth labels, trading requires optimizing a portfolio-level objective that accounts for both return and risk. Computing gradients of the Sharpe ratio required careful handling of the mean and standard deviation. This gradient encourages the model to increase positions when returns are above average (positive contribution to mu_R) and reduce positions during high-volatility periods (to reduce sigma_R).

While Sharpe ratio was my primary objective, I also implemented several alternative losses for different investor preferences:
  - Sortino Ratio Loss: Only penalizes downside volatility
  - Calmar Ratio Loss: Ratio of return to maximum drawdown
  - Sharpe with Turnover Penalty: Incorporates transaction costs direclty into the loss

As for optimization, I used AdamW (Adam with weight decay) as my optimization function. The weight decay term is crucial for preventing overfitting in financial models, which often have low signal-to-noise ratios. I also introduced gradient clipping to prevent any exploding gradients. As well as this, I implemented a ReduceLROnPlateau scheduling, which reduces the learning rate when validation Sharpe ratio pleateaus. This adaptive schedule allows the model to take large steps initially then fine-tune as it converges. 

## Feature Engineering
One of the most critical components of my system is the feature engineering pipeline. Financial time series requires careful preprocessing to extract meaningful, generalizable features.

### The Challenge of Stock-Agnostic Features
Traditional momentum strategies compute features like "20-day moving average" or "daily volume" directly from OHCLV data. However, these raw features are fundamentally incomparable across different stocks. 
Example:
  - Apple (AAPL): Price ~$180, Volume ~50M shares/day
  - Berkshire Hathaway Class A (BRKA): Price ~$500,000, Volume ~1,000 shares/day
  - Penny stock: Price ~$0.50, Volume ~5M shares/day
The model has no way to know that "$500,000 per share" and "$180 per share" both represent large-cap blue-chip stocks, just with nominal prices.

This scale problem is futher complicated by distributional differences. Some stocks have very stable, low volatility returns. Others exhibit extreme volatility. A fixed model architecture cannot adapt to these diverse characteristics. 

### Stock-Agnostic Design Principles
To enable generalization, I designed features according to three core principles:
  1. Dimensionless Ratios: Instead of using absolute values, I computed ratios that are meaningful regardless of scale
  2. Normalized Statistics: Standardized features relative to each stock's own history
  3. Technical Indicators with Fixed Ranges: Included indicators that naturally ahve bounded outputs

### Complete Feature Taxonomy
My feature set comprises 30-35 features (depending on the data availability) organized into several categories:
  - Returns-Based Features (4-6 features)
    - Returns are normally stock-agnostic because they represent percentage changes, which are comparable across assets. Multi-horizon windows allow the model to distinguish between different momentum patterns: a stock might have strong 1-week momentum but weak 3-month momentum, suggesting mean-reversion may be imminent.
  - Price Ratio Features (5-7 features)
    - Rather than using absolute prices, I computed deviations from moving averages. These ratios capture price momentum in a scale-invariant way. A 5% deviation from the 50-day MA has the same meaning whether the stock price is $10 or $10,000.
  - Volatility Features (4-6 features)
    - Volatility is measured as the standard deviation of returns (already dimensionless). Volatility ratios help identify regime changes: a vol_ratio > 1.5 suggests volatility is spiking, which often precedes mean-reversion.
  - Technical Indicators (3-4 features)
    - I included standard technical indicators that have natural bounded ranges. These indicators are interpreable: RSI > 70 suggests overbought conditions regardless of the specific stock, while bb_position = 1 means the price is at the upper Bollinger Band
  - Volume Features (3-4 features)
    - Volume is tricky because absolute volume is meaningless. I normalized relative to each stock's typical volume. A volume_ratio of 3.0 means today's volume is 3x the 20-day average -- this is meaningful whether the absolute volume is 100K or 100M shares.
  - Statistical Moments (4 features)
    - Higher-order moments capture distributional properties. These moments help the model recognize dangerous regimes: negative skewness indicates a distribution with fat left tails (large losses more common than large gains)
  - Autocorrelation Features (3 features)
    - Autocorrelation measures how much current returns predict future returns. Positive autocorrelation (>0.3) indicates trending behavior where momentum works well. Negative autocorrelation suggest mean-reversion where momentum fails
  - Z-score Normalization
    - For features without natural bounds, I standardized relative to history. These z-scores answer "how unusual is the current value?" A price_zscore of 2.0 means that the price is 2 standard deviations above its yearly mean -- regardless of whether the stock is at $10 or $1000
  - Market Context Features (5 features, if SPY data is available)
    - Perhaps the most important features for multi-asset trading are those that capture market context. These features are crucial because a stock's behavior should be interpreted in market context. A -2% return is bad in isolation, but if the market dropped -5%, it represents relative strength.

# Training Methodology
## Data Preparation and Quality Control
The quality of training data ultimately determines model performance.

### Data Source and Format
My raw data consisted of hourly OHLCV bars for 100+ US equities and ETFs spanning from 2015-2025. Each bar contained:
  - Timestamp (UTC)
  - Open, High, Low, Close prices
  - Volume (number of shares traded)
  - Symbol identifier
the hourly frequency balances two competing objectives: 1. Sufficient data points for training the deep network, and 2. low enough frequency to avoid microstructure noise that dominates high-frequency data. Hourly bars provided approximately 6,000 observations per symbol per year, sufficient for training the models with ~100K parameters.

### Multi-Stage Filtering Pipeline
To ensure robust out-of-sample performance, I implemented aggressive filtering to remove problematic securities. The filtering proceeds in four stages, each eliminating different failure modes.
  1. Data Quality Requirements: The first stage ensures sufficient data exists for meaningful training. Securities with sparse data prevent proper train/validation/test splits. Penny stocks (<$2) often have artificial volatility due to minimum tick sizes (e.g., a $0.50 stock moving to #0.51 is a 2% return, but this isn't a meaningful signal). Low volume securities have wide bid-ask spreads making realistic backtesting incredibly difficult. Volume consistency filters out securities that trade sporadically.
  2. Leverage Product Exclusion: The second stage removes leveraged and inverse ETFs, which exhibit properties fundamentally different from normal securities. Why did I exclude leveraged products?
     - Leveraged ETFs use derivatives to amplify returns, which creates several problems:
       1. Volatility Decay: Due to daily rebalancing, leveraged ETFs underperform even when correctly predicting direction. A 3x bull ETF will not return 3x the index over time, it will typically return less than 3x due to volatility drag. This breaks the assumption that accurate predictios generate proportional profits.
       2. Path Dependency: Returns depend on the path taken, not just start/end points. This violates the i.i.d assumption underlying our statistical models.
       3. Unrealistic Volatility: 3x ETFs can have annualized volatility exceeding 100%, which is unrepresentative of normal securities and leads models to overfit to extreme movements.
       4. Funding Costs: Leveraged products incur daily financing charges that aren't captured in simple price data, leading to systematic bias in backtesting results.
  3. Statistical Validation: The third stage examines return distributions for pathological properties. I computed various distribution statistics and flag securities that have excessive outliers, have extreme positive/negative drift, and trade infrequently.
  4. Correlation and Diversification: This prevents the portfolio fro becoming concentrated in highly correlated securities (e.g., selecting 10 different semiconductor stocks), which would increase idosyncratic risk without improving diversification.

### Final Characteristics
After filtering, the universe comprised of 52 securities with the following poperties:

- Sector Diversification:
  - Technology: 12 securities (AAPL, MSFT, NVDA, AMD, CSCO, HPQ, ...)
  - Consumer Discretionary: 8 securities (KDP, CMCSA, NFLX, DIS, ...)
  - Financials: 6 securities (BAC, JPM, GS, ...)
  - Industrials: 4 securities (BA, CAT, ...)
  - Internationals: 6 securities (EWZ, EWW, EWH, ...)
  - Commodities: 5 securities (VALE, FCX, ...)
  - Other: 11 securities
- Market Cap Distribution:
  - Large Cap (>$100B): 18 securities
  - Mid Cap ($10B - $100B): 24 securities
  - Small Cap ($1B - $10B): 10 securities
- Volatility Distribution:
  - Low Vol (<25% annual): 8 securities
  - Medium Vol (25-50% annual): 32 securities
  - High Vol (>50% annual): 12 securities
- Beta Distribution:
  - Defensive (Beta < 0.8): 10 securities
  - Neutral (0.8 <= Beta <= 1.2): 28 securities
  - Aggressive: (Beta > 1.2): 14 securities

### Train/Validation/Test Split Strategy
Proper data splitting is crucial for obtaining realistic performance estimates. Unlike i.i.d. data where random splits suffice, time series data requires chronological splits to prevent lookahead bias. 

### Time-Based Splitting
<img width="697" height="90" alt="Screenshot 2026-01-27 at 8 12 32 PM" src="https://github.com/user-attachments/assets/3e051615-7fff-4792-a5e2-559f63570b56" />

The 60%/20%/20% split was designed to:
  1. Sufficient Training Data: 6 years captures multiple market regimes (2015-2016 bull market, 2018 correction, 2020 COVID crash, 2020-2021 recovery). This diversity prevents the model from learning regime-specific patterns.
  2. Validation for Early Stopping: 2 years provides enough data to reliably estimate validation Sharpe ratio (12,000 observations -> standard error = 0.0091 for Sharpe). This enables meaningful model selection without excessive variance.
  3. Out-of-Sample Test: 23 months represents a genuine forward test, including the 2023 banking crisis, rate hike cycle, and 2024 market dynamics. Using 2023-2024 as test ensures no information from the current market environment leaked into the training.

### Walk-Forward Validation
Beyond the simple train/val/test split, I implemented walk-forward analysis on the test set to assess stability:
<img width="701" height="134" alt="Screenshot 2026-01-27 at 8 16 27 PM" src="https://github.com/user-attachments/assets/19194204-e847-45bc-8421-6ca2a905d289" />

Walk-Forward Parameters:
  - Training window: 3 years (756 days x 24 hours = 18,144 observations)
  - Test window: 1 quarter (63 days x 24 hours = 1,512 observations)
  - Step size: 1 month (21 days x 24 hours = 504 observations)
This generated approximately 20-25 separate train/test windows, allowing to compute confidence intervals on performance metrics. If a strategy showed high Sharpe in one window but fails in others, it suggests overfitting to specific market conditions.

I also implemented an expanding window where the training start stays fixed but the end moves forward. This differs from a rolling window that would maintain constant length. The expanding approach accumulates more diverse training data over time, better represents realistic deployment, and avoids sudden performance drops when old regimes exit the rolling window. 

## Model Training Procedure
Training proceeds in two phases: 
  1. Component model training (vanilla and attention models separately)
  2. Ensemble training with all parameters jointly optimized.

### Phase 1: Training Component Models
I first trained the vanilla and attention-enhanced models independently on the training set. The key insight is that I computed Sharpe ratio at the epoch level, not per-batch. Batch-level Sharpe estimates are extremely noise (with high variance), so I accumulated all predictions from the epoch and computed a single Sharpe value. This provides more stable gradient signals. 

After each epoch, I evaluated on the validation set, where early stopping tracks the best validation Sharpe ratio. 

### Phase 2: Training the Ensemble
Once the component models were trained, I initialized the ensemble with pre-trained weights. I then expirimented with two training approaches:
  1. Freeze Base Models, Train Weight Network Only: This approach was safer as it prevents destroying the pre-trained represenations, however it is less flexible.
  2. End-to-End Fine-Tuning: This approach allowed component models to adjust their predictions to complement each other, potentially discovering dynergies. However, it could risk catastrophic forgetting if not carefully tuned.
I discovered that the second approach achieved superior performance with a Test Sharpe of 1.87.

# Backtesting and Performance Analysis
## Backtesting Framework
Realistic backtesting is perhaps the most critical, and most commonly mishandled, aspect of trading system development. Many academic papers and online tutorials present backtests with unrealistic assumptions that lead to inflated performance estimates. I hope to explain my comprehensive backtesting framework designed to provide comprehensive, implementable performance estimates. 

### Transaction Cost Modeling
Real world trading incurs several types of costs that must be modeled for realitic performance assessment:
  - Bid-Ask Spread: The difference between the price at which you can sell (bid) and buy (ask). For liquid securities during normal hours, spreads are typically 1-3 basis points. For less liquid securities or duing volatile periods, spreads can widen to 10-50 basis points.
  - Market Impact: Large orders move prices against you. My portfolio is relatively small (~10 positions), so I assume minimal market impact.
  - Slippage: The difference between expected and actual execution price due to timing delays. I modeled this as an additional cost proportional to position change.

### Portfolio Construction and Rebalancing
Unlike single-stock prediction, portfolio trading requires rules for position sizing, diversificaiton, and rebalancing. My framework implement these realistic constraints. 
- Universe Selection: At each rebalancing point, I have predictions for all 52 securities in my universe. Then construct a long-short portfolio. Here are the portfolio characteristics:
  - Total long exposure: 50% (5 positions x ~10% each)
  - Total short exposure: -50% (5 positions x -10% each)
  - Net exposure: ~0% (dollar-neutral)
  - Gross exposure: 100% (leverage ratio of 1.0)

I also implemented a weekly rebalance with a minimum threshold. The weekly rebalancing balances signal freshness with transaction costs. Daily rebalancing would incorporate new information faster but incur excessive costs. Monthly rebalancing reduces costs but allows positions to drift too far from optimal. The 1% threshold prevents tiny adjustments that cost more in commissions than they gain in optimization. 

### Walk-Forward Validation
My primary backtesting methodology was walk-forward analysis, which simulated realistic model retraining. My choice of 3-year training windows and 1-quarter test windows was motivated by:
  1. Sufficient Training Data: 3 years x 252 days/year x 6 hours/day = ~4,500 timesteps per symbol. With 52 symbols this provides ~234,000 training samples, sufficient for a model with ~100K parameters without severe overfitting.
  2. Regime Coverage: 3 years typically captures multiple market regimes (bull market, correction, recovery), ensuring the model learns robust patterns rather than regime-specific behaviors.
  3. Realistic Test Period: 1 quarter (3 months) represents a reasonable investment horizon for momentum strategies. Shorter periods have excessive noise; longer periods allow substantial drift in market dynamics
  4. Computational Efficiency: Retraining models is expensive. Monthly rolling (step_size = 1 month) provides a good balance between keeping models fresh and computational cost

With N = 20-25 windows, I constructed approximate confidence intervlas for Sharpe ratio. This demonstrated the strategy's Sharpe ratio is consistently strong across different market periods, not just lucky on a single period. 

## Results and Key Findings
My comprehensive backtest reveals several important insights about model performance, regime dependencies, and the value of ensemble approaches

### Overall Performance (Regime-Aware Ensemble)
Testing on the period January 2023 to November 2024 (23 months, 12,307 hourly observations, the regime aware ensemble was able to achieve: 
<img width="395" height="268" alt="Screenshot 2026-01-27 at 8 41 13 PM" src="https://github.com/user-attachments/assets/7306784d-eda0-4908-a829-e97850912cf7" />

The 1.87 Sharpe ratio is exceptional for a quantitative equity strategy: For some context:
  - Market (SPY): Historical Sharpe ~0.4-0.5
  - Typical momentum strategy: Sharpe ~0.6-1.0
  - Top-quartile hedge fund: Sharpe ~1.0-1.5
  - Renaissance Medallion: Sharpe ~2.0+

The 31% volatility is higher than long-only equities (~15-20%) due to levearge (100% gross exposure) and short positions. However, it's well below leveraged products (50-100% vol for 2-3x ETFs), indicating solid risk management. Even under pessimistic cost assumptions (20 bps), the strategy maintains a Sharpe ratio of 1.72, indicating genuine alpha that survives realistic trading frictions. 

### Model Ablation Study
To validate the importance of the regime-aware ensemble, I compared the three configurations
<img width="704" height="153" alt="Screenshot 2026-01-27 at 8 50 01 PM" src="https://github.com/user-attachments/assets/f0590389-25d1-4f4d-9b68-c4ba21f4da28" />

Critical Findings:
  - Vanilla Model Alone Fails: Achieving nearly zero return (0.092 Sharpe) demonstrates that simple transformers cannot adapt to changing market conditions. Performance degrades severly during the Q1 2023 banking crisis and subsequent volatility.
  - Attention Model Alone Overfits: Negative returns (-21.44%) indicate the attention model learned patterns specific to the training period that failed to generalize. The added model capacity hurt more than it helped when used in isolation.
  - Adaptive Ensemble Dominates: Dynamic weighting based on regime detection improves returns to 198% over the testing horizon and Sharpe to 1.87.

### Feature Importance and Model Intepretation
Understanding which features drive predictions is crucial for validating that th emodel learns economially meaningful patterns rather than spurious correlations. I analyzed the transformer's attention weights to determine which timesteps the model focuses on:

Findings:
  - Recency Bias: The model attends most heavily to the last 20-60 timesteps (hours), consistent with momentum having a 1-day to 1-week horizon
  - Volatility Spike Attention: During periods of high volatility, attention concentrates on the most recent 5-10 hours, suggesting the model recognizes urgent information
  - Periodic Patterns: Some attention heads show weekly periodicity (every 168 hours = 1 week), potentially capturing day-of-week effects
  - Regime Transitions: During regime changes (e.g., going from low to high vol), the model attends to the last similar regime transition, suggesting it learns to identify analogous situations.

The top features align with momentum theory:
  1. Recent Returns: Past performance predicts future performance (momentum effect)
  2. Volatility: High vol reduces momentum effectiveness (requires position sizing)
  3. RSI/MACD: Classic technical indicators capture overbought/oversold
  4. Market Context: Individual stock behavior depends on market environment
  5. Relative Strength: Stocks outperforming the market tend to continue.

### Information Coefficient Analysis
The Information Coefficient (IC) measures the correlation between predictions and realized returns. IC is a standard metric for assessing prediction quality. Unlike accuracy (which is binary), IC measures the strength of the relationship. 
<img width="500" height="143" alt="Screenshot 2026-01-27 at 8 59 15 PM" src="https://github.com/user-attachments/assets/3a20f711-5b8e-4df0-b155-8539f5f9bbd7" />

An IC of 0.0098 means predictions and returns have a 0.0098 correlation. While this seems small, several factors make it meaningful:
  1. Statistical Significance: the p-value of 0.010 indicates that the IC is significantly different from zero at the 99% confidence level.
  2. Hourly Predictions: Hourly returns are extremely noise. Even weak correlations are difficult to achieve and are valuable if consistent
  3. Actionable Signal: An IC of 0.01 can generate substantial profits if applied consistently across many securities over long periods. They key is the signal persists out-of-sample
  4. Top Quartile Range: Published research on momentum strategies reports ICs of 0.005 - 0.015 for similar prediciton horizons.


# Use Cases and Applications
The regime-aware momentum transformer is designed for practical deployment across various institutional and retail contexts. 

### Multi-Asset Portfolio Management
An investment manager overseeing a $10M - $100M portfolio seeks systematic diversification across equities, sectors, and geographies while maintaining market neutrality.

### Hedge Fund Strategy
A quantitative hedge fund running a long-short equity book with $100M+ AUM, seeking uncorrelated returns to traditional risk premia. 
Advantages:
  - Market Neutral Returns: Net exposure near zero provides true alpha (low correlation to SPY: 0.12)
  - Capacity: Weekly rebalancing and liquid universe supports $100M - $500M AUM without significant market impact
  - Risk Adjusted Performance: 1.87 Sharpe at fund level is top-quartile for equity strategies
  - Transparency: Model-based approach provides clear attribution and risk decomposition

### Algorithmic Trading System
A proprietary trading desk or sophisticated retail trader seeks to automate momentum trading across dozens of securities with hourly rebalancing.

### Research Platform
Academic researchers or quantitative analysts seeking to study momentum strategies, regime detection, and ensemble methods. 

# Installation and Usage
## System Requirements
Hardware:
  - CPU: Modern multi-core processor (Intel i5/i7/i9 or AMD equivalent)
  - RAM: 16GB minimum, 32GB recommended for training
  - Storage: 50GB for data + models
  - GPU: Optional but recommended

Softare:
  - Python 3.8+
  - Cuda 11.0+ (if using GPU)
  - Operatig System: Linux (Ubuntu 20.04+), macOS 11+, or Windows 10+

## Installation
Step 1: Clone Repository
git clone https://github.com/your-username/momentum-transformer.git
cd momentum-transformer

Step 2: Create Virtual Environment
#using venv
python -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
#Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install numpy pandas scikit-learn matplotlib seaborn tqdm

# Optional: Ray Tune for hyperparameter optimization
pip install "ray[tune]" hyperopt bayesian-optimization

# Optional: HMM for advanced regime detection
pip install hmmlearn

# Optional: Alpaca API for live trading
pip install alpaca-trade-api

Step 4: Complete requirements.txt
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

# Optional dependencies
ray[tune]>=2.9.0         # Hyperparameter tuning
hyperopt>=0.2.7
bayesian-optimization>=1.4.0
hmmlearn>=0.3.0          # HMM regime detection
alpaca-trade-api>=3.0.0  # Live trading

# Conclusion
This project demonstrated that sophisticated deep learning architecture can be successfully applied to financial time series prediction when combined with careful feature engineering, realistic backtesting, and adaptive model selection. The regime-awware ensemble achieves a Sharpe ratio of 1.87 on out-of-sample data, placing it among the top-performing quantitative equity strategies. 

Several key insights emerged from this project. 

1. Generalization Requires Stock-Agnostic Features: The single most important contribution was the development of normalized, dimensionless features that work across different securities. Without this foundation, even sophisticated neural architectures fail to generalize beyond their training distribution.
2. No Single Architecture Dominates: My ablation studies definitively show that neither vanilla nor attention-enhanced transformers work well across all market regimes. The vanilla model excels in stable trending markets but fails during volatility spikes. The attention model captures regime changes but overfits during normal periods. Only an adaptive ensemble that dynamically weights models based on detected market conditions achieves consistently strong performance.
3. Interpretability Validates Economic Reasoning: Analysis of feature importance and attention weights reveal that the model learns patterns consistent with momentum theory: it focuses on recent returns, volatility regimes, and relative strength. This interpretability provides confidence that the model captures genuine market dynamics rather than spurious correlations.





