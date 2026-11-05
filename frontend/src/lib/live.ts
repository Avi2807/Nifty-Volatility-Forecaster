/** True once the session a forecast was for has ended, at 15:30 in Mumbai, wherever the viewer is. */
export function sessionHasClosed(targetDate: string, now: Date): boolean {
  return now.getTime() >= Date.parse(`${targetDate}T15:30:00+05:30`);
}
