/** The one place log realized variance is converted into the units people read. */

export const TRADING_DAYS = 252;

/** Annualised volatility in percent: sqrt(exp(x) * 252) * 100. */
export function logRvToAnnualVolPct(logRv: number): number {
  return Math.sqrt(Math.exp(logRv) * TRADING_DAYS) * 100;
}

/** One standard deviation of the day's open-to-close return, in percent: sqrt(exp(x)) * 100. */
export function logRvToDailyMovePct(logRv: number): number {
  return Math.sqrt(Math.exp(logRv)) * 100;
}
