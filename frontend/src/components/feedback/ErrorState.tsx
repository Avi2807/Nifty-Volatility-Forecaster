import { API_URL, ApiError, USE_MOCKS } from "../../api/client";
import styles from "./Feedback.module.css";

const START_COMMAND = ".venv\\Scripts\\python -m uvicorn src.api.main:app --port 8000";

interface ErrorStateProps {
  error: unknown;
  title?: string;
  onRetry?: () => void;
}

/** A backend that can't be reached gets start-up help. A request it refused shows the backend's message. */
export function ErrorState({ error, title = "This section couldn't load", onRetry }: ErrorStateProps) {
  const unreachable = !(error instanceof ApiError) || error.status === undefined;
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className={styles.error} role="alert">
      <p className={styles.errorTitle}>
        <span aria-hidden="true">✕ </span>
        {title}
      </p>
      <p>{message}</p>
      {unreachable && !USE_MOCKS && (
        <>
          <p>
            The app expects the backend at <code>{API_URL}</code>. Start it from the project root:
          </p>
          <pre className={styles.command}>
            <code>{START_COMMAND}</code>
          </pre>
          <p>
            If it says artifacts are missing, run <code>.venv\Scripts\python run_all.py</code> first. To work without a backend, run{" "}
            <code>npm run dev:mock</code>.
          </p>
        </>
      )}
      {onRetry && (
        <button type="button" className={styles.retry} onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}
