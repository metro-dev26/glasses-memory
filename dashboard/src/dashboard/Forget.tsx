import { useEffect, useState } from "react";

const CONFIRM_MS = 6000; // the confirm step closes itself if not used

/** "Forget everything", with the confirmation inside the page. */
export function Forget({ onForget, disabled }: { onForget: () => void; disabled: boolean }) {
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    if (!confirming) return;
    const t = window.setTimeout(() => setConfirming(false), CONFIRM_MS);
    return () => window.clearTimeout(t);
  }, [confirming]);

  if (!confirming) {
    return <button className="ghost small danger-text" disabled={disabled} onClick={() => setConfirming(true)}>Forget everything</button>;
  }
  return (
    <span className="confirm">
      <span>Delete every object, fingerprint and snapshot?</span>
      <button className="danger small" onClick={() => { setConfirming(false); onForget(); }}>Forget</button>
      <button className="ghost small" onClick={() => setConfirming(false)}>Cancel</button>
    </span>
  );
}
