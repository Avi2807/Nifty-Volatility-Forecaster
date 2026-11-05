import { useEffect, useState } from "react";

/** Goes up when anything a chart draws with changes: the theme attribute, or the web fonts arriving. */
function useDrawingVersion(): number {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const bump = () => setVersion((current) => current + 1);
    const observer = new MutationObserver(bump);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    let active = true;
    void document.fonts?.ready.then(() => {
      if (active) bump();
    });
    return () => {
      active = false;
      observer.disconnect();
    };
  }, []);
  return version;
}

/**
 * Canvas charts and d3 scales need real colour values, not var(--name). This reads them from
 * tokens.css and rereads them when the theme changes, so charts restyle with the rest of the page.
 */
export function useTokenReader(): (token: string) => string {
  const version = useDrawingVersion();
  const [reader, setReader] = useState(() => makeReader());
  useEffect(() => setReader(() => makeReader()), [version]);
  return reader;
}

function makeReader() {
  const styles = getComputedStyle(document.documentElement);
  return (token: string) => styles.getPropertyValue(token).trim();
}
