/** "just now", "42s ago", "3m ago", "2h ago" from a Unix time in seconds. */
export function ago(unixSeconds: number, now = Date.now() / 1000): string {
  return agoSeconds(now - unixSeconds);
}

export function agoSeconds(s: number): string {
  s = Math.max(0, Math.round(s));
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

export function clock(unixSeconds: number): string {
  return new Date(unixSeconds * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
}

export function elapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
