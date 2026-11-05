# Model comparison: the verdict

Test period: 615 forecasts for sessions from 2021-01-04 to 2023-07-03. Lower is better on both scores. "The Transformer" means the with-news model unless it says otherwise.

## Where the Transformer ranks

On MSE on log RV it ranks 2nd of 8, behind the no-news Transformer.

On QLIKE it ranks 1st of 8.

**It loses to a simpler model on: the no-news Transformer on MSE.**

One seed at a time, the mean MSE runs in this order: LSTM 0.2581, the no-news Transformer 0.2654, MLP 0.2782, Transformer 0.2841. One seed at a time, the mean QLIKE runs in this order: LSTM 0.1493, Transformer 0.1560, the no-news Transformer 0.1562, MLP 0.1577. The Transformer's MSE varies by 0.0236 between seeds, more than its gap to MLP, LSTM and the no-news Transformer, so those positions could change with other seeds. The Diebold-Mariano tests treat each averaged forecast as fixed, so they don't include this.

## Is each lead statistically significant?

Diebold-Mariano tests with 5 Newey-West lags. A negative statistic means the Transformer had the lower loss. These are 14 tests, so a Bonferroni limit of 0.0036 applies to them as a family.

| Against | MSE statistic | MSE p | Lower MSE | QLIKE statistic | QLIKE p | Lower QLIKE |
|---|---|---|---|---|---|---|
| Naive | -5.89 | <0.0001 | Transformer | -6.31 | <0.0001 | Transformer |
| VIX-implied | -18.55 | <0.0001 | Transformer | -15.91 | <0.0001 | Transformer |
| HAR-RV | -2.24 | 0.0250 | Transformer | -2.89 | 0.0038 | Transformer |
| GARCH(1,1) | -9.38 | <0.0001 | Transformer | -6.08 | <0.0001 | Transformer |
| MLP | -0.70 | 0.4861 | Transformer | -1.44 | 0.1503 | Transformer |
| LSTM | -0.26 | 0.7959 | Transformer | -1.84 | 0.0657 | Transformer |
| Transformer, no news | 0.12 | 0.9010 | Transformer, no news | -1.91 | 0.0558 | Transformer |

In words:

- **Naive.** On MSE the Transformer is ahead of Naive, and the lead holds up after the Bonferroni adjustment. On QLIKE it is ahead of Naive, and the lead holds up after the Bonferroni adjustment.
- **VIX-implied.** On MSE the Transformer is ahead of VIX-implied, and the lead holds up after the Bonferroni adjustment. On QLIKE it is ahead of VIX-implied, and the lead holds up after the Bonferroni adjustment.
- **HAR-RV.** On MSE the Transformer is ahead of HAR-RV, and the lead is significant at 5% but not after the Bonferroni adjustment. On QLIKE it is ahead of HAR-RV, and the lead is significant at 5% but not after the Bonferroni adjustment.
- **GARCH(1,1).** On MSE the Transformer is ahead of GARCH(1,1), and the lead holds up after the Bonferroni adjustment. On QLIKE it is ahead of GARCH(1,1), and the lead holds up after the Bonferroni adjustment.
- **MLP.** On MSE the Transformer is ahead of MLP, and the lead isn't statistically significant. On QLIKE it is ahead of MLP, and the lead isn't statistically significant.
- **LSTM.** On MSE the Transformer is ahead of LSTM, and the lead isn't statistically significant. On QLIKE it is ahead of LSTM, and the lead isn't statistically significant.
- **Transformer, no news.** On MSE the Transformer is behind the no-news Transformer, though the gap isn't statistically significant. On QLIKE it is ahead of the no-news Transformer, and the lead isn't statistically significant.

## When it does better or worse

Test days are sorted into three equal groups by the 22-day average of log RV known when the forecast was made, so the grouping can't favour a model that happened to forecast high or low.

| Period | Days | 22-day volatility, annualised | Transformer MSE (rank) | HAR MSE | p vs HAR | Transformer QLIKE (rank) | HAR QLIKE | p vs HAR | Best on QLIKE |
|---|---|---|---|---|---|---|---|---|---|
| Calm | 205 | 5.6% to 8.6% | 0.2395 (2) | 0.2715 | 0.0450 | 0.1356 (2) | 0.1487 | 0.2766 | MLP |
| Middle | 205 | 8.6% to 11.0% | 0.2644 (3) | 0.2709 | 0.7831 | 0.1463 (1) | 0.1740 | 0.1378 | Transformer |
| Volatile | 205 | 11.0% to 15.0% | 0.2351 (3) | 0.2690 | 0.0147 | 0.1195 (1) | 0.1488 | 0.0003 | Transformer |

Against HAR, the Transformer's QLIKE gap is +0.0130 in calm periods and +0.0292 in volatile ones, where a positive gap means the Transformer is ahead. Across rolling 63-day windows it's ahead of HAR in 72% of them. Its largest lead is 0.076 in the window ending 2023-03-16, and its largest deficit is 0.030 in the window ending 2021-08-18. Day by day it's ahead of HAR on QLIKE only 50% of the time, and its 10 best days supply 59% of its lead (0.0233 a day with them, 0.0098 without). So the lead comes mainly from a few sessions, the largest on 2023-01-30 and 2023-02-01, not from being a little better every day. These groups are small and were split after the fact, so treat them as description, not proof. "Volatile" is relative to this period too: the 22-day figure tops out at 15.0% annualised, because the 2020 crash fell in the validation period. So this says little about how any model would do in a crisis.

## 99% VaR

The same rule for every model: 2.33 x sqrt(exp(forecast)). A sound VaR would breach about 6.2 times. Kupiec fails above 3.84.

| Model | Breaches | Kupiec LR | Result |
|---|---|---|---|
| Naive | 26 | 35.92 | fail |
| VIX-implied | 0 | 12.36 | fail |
| HAR-RV | 26 | 35.92 | fail |
| GARCH(1,1) | 8 | 0.51 | pass |
| MLP | 25 | 33.01 | fail |
| LSTM | 24 | 30.19 | fail |
| Transformer, no news | 28 | 41.98 | fail |
| Transformer | 24 | 30.19 | fail |

Passing: GARCH(1,1). A forecast of RV runs low for this purpose however accurate it is, because the squared open-to-close return averages 1.31 times RV. GARCH targets the return's own variance, and its forecasts sit +0.472 log units from log RV on average, which is why its line is crossed less often. A model that passes here isn't necessarily the better volatility forecaster: VaR only counts how often one line was crossed.

## All the scores

For the neural models, the score is that of the forecast averaged over 3 seeds, and "seed mean ± spread" is each seed's own score, with the sample standard deviation.

| Model | MSE | MSE, seed mean ± spread | QLIKE | QLIKE, seed mean ± spread | Parameters | Training time |
|---|---|---|---|---|---|---|
| Naive | 0.3609 | n/a | 0.2042 | n/a | 0 | nothing to fit |
| VIX-implied | 1.6748 | n/a | 0.5352 | n/a | 0 | nothing to fit |
| HAR-RV | 0.2705 | n/a | 0.1571 | n/a | 4 | under 0.001s |
| GARCH(1,1) | 0.5443 | n/a | 0.2144 | n/a | 3 | 0.301s |
| MLP | 0.2536 | 0.2782 ± 0.0122 | 0.1438 | 0.1577 ± 0.0124 | 7,937 | 0.723s |
| LSTM | 0.2485 | 0.2581 ± 0.0108 | 0.1434 | 0.1493 ± 0.0146 | 6,561 | 45.4s |
| Transformer, no news | 0.2453 | 0.2654 ± 0.0220 | 0.1437 | 0.1562 ± 0.0188 | 9,665 | 31.2s |
| Transformer | 0.2463 | 0.2841 ± 0.0236 | 0.1338 | 0.1560 ± 0.0133 | 9,761 | 37.2s |

Added training time for GARCH, the MLP and the LSTM: 46.4s.

## Read this before relying on it

- **The training settings aren't identical.** The Transformer's were chosen on validation data: learning rate 5e-4, weight decay 1e-3, a 5-epoch warm-up, and last-day pooling after mean pooling did poorly. The MLP and LSTM use the settings the brief gave (learning rate 0.001, weight decay 0.0001, no warm-up) with no tuning, as it asked. A tuned MLP or LSTM might do better, so this tests them as specified, not at their best.
- **GARCH is scored on a target it doesn't model.** It forecasts the variance of the open-to-close return, not log RV. Its forecasts average +0.472 log units from log RV, and without that average gap its MSE would be 0.3213 rather than 0.5443. That figure only describes the gap, because no forecast could have known it in advance.
- **The MLP and LSTM use 18 features, not the brief's 17,** following the project's earlier decision on the weekday columns. So the MLP has 90 inputs rather than 85. They use the news features, like the with-news Transformer.
- **The Transformer's test scores were already known** when these models were added. Nothing about the new models was chosen after seeing test results, and their checkpoints were checksummed before any test forecast was made.
- **Unadjusted p-values overstate the evidence** when many tests are run, which is why the Bonferroni limit is given.
- **The later check isn't extended.** These models were only scored on the test period, so nothing here says how they did after June 2023.
