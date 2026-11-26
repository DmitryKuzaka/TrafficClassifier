import { useQuery } from "@tanstack/react-query";

import { getModelReadiness } from "./api/client";
import styles from "./ModelStatus.module.css";

export function ModelStatus() {
  const readiness = useQuery({
    queryKey: ["model-readiness"],
    queryFn: ({ signal }) => getModelReadiness(signal),
    retry: false,
  });

  const label = readiness.isError
    ? "Модель недоступна"
    : readiness.isPending
      ? "Проверяем модель"
      : "Модель готова";

  return (
    <>
      <div className={styles.status} data-unavailable={readiness.isError}>
        <span className={styles.label} data-state={readiness.status} role="status">
          <span className={styles.dot} aria-hidden="true" />
          {label}
        </span>
        {readiness.isError && (
          <button
            className={styles.retry}
            type="button"
            aria-describedby="model-readiness-error"
            disabled={readiness.isFetching}
            onClick={() => void readiness.refetch()}
          >
            {readiness.isFetching ? "Проверяем…" : "Повторить"}
          </button>
        )}
      </div>
      {readiness.isError && (
        <p className={styles.notice} id="model-readiness-error" role="alert">
          {readiness.error.message}
        </p>
      )}
    </>
  );
}
