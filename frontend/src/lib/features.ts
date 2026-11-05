/** How each model input is named and written on screen. Weekday columns are shown as a caption instead. */

export interface FeatureDisplay {
  name: string;
  label: string;
  format: (value: number) => string;
}

const withMinus = (text: string) => text.replace("-", "\u2212");
const number2 = (value: number) => withMinus(value.toFixed(2));
const number3 = (value: number) => withMinus(value.toFixed(3));
const percent = (value: number) => withMinus(`${(value * 100).toFixed(2)}%`);
const scientific = (value: number) => value.toExponential(2);
const share = (value: number) => `${(value * 100).toFixed(1)}%`;
const yesNo = (value: number) => (value === 1 ? "Yes" : value === 0 ? "No" : `${(value * 100).toFixed(0)}% of days`);

export const MARKET_DISPLAY: FeatureDisplay[] = [
  { name: "log_rv", label: "Log realized variance, today", format: number2 },
  { name: "log_rv_w", label: "Log RV, 5-day mean", format: number2 },
  { name: "log_rv_m", label: "Log RV, 22-day mean", format: number2 },
  { name: "parkinson", label: "Parkinson range variance", format: scientific },
  { name: "ret", label: "Daily return", format: percent },
  { name: "ret_neg", label: "Downside return", format: percent },
  { name: "gap", label: "Overnight gap", format: percent },
  { name: "log_vix", label: "Log India VIX", format: number3 },
  { name: "log_turnover_ratio", label: "Log turnover vs 20-day mean", format: number3 },
  { name: "expiry_next", label: "Next session is an expiry", format: yesNo },
];

export const NEWS_DISPLAY: FeatureDisplay[] = [
  { name: "news_mean_sentiment", label: "News sentiment, mean", format: number3 },
  { name: "news_negative_share", label: "Negative headline share", format: share },
  { name: "news_log_count", label: "Log (1 + headline count)", format: number2 },
];

/** How far a value sits from its train-period mean, in standard deviations. */
export function standardScore(value: number, mean: number, std: number): number | null {
  return std > 0 ? (value - mean) / std : null;
}
