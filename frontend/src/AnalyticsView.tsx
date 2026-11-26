import { skipToken, useQuery } from "@tanstack/react-query";

import { getCompletedRuns, getRunStatistics } from "./api/client";
import { RunSummary } from "./RunSummary";
import styles from "./AnalyticsView.module.css";

const utcDate = new Intl.DateTimeFormat("ru-RU", {
  dateStyle: "short",
  timeStyle: "medium",
  timeZone: "UTC",
});
const integer = new Intl.NumberFormat("ru-RU");

type Props = {
  runId: string | null;
  onRunChange: (runId: string) => void;
};

export function AnalyticsView({ runId, onRunChange }: Props) {
  const runs = useQuery({
    queryKey: ["completed-runs"],
    queryFn: ({ signal }) => getCompletedRuns(signal),
    networkMode: "always",
    retry: false,
  });
  const active = runs.data?.find((run) => run.run_id === runId) ?? runs.data?.[0];
  const statistics = useQuery({
    queryKey: ["run-statistics", active?.run_id],
    queryFn: active ? ({ signal }) => getRunStatistics(active.run_id, signal) : skipToken,
    enabled: !runs.isError && Boolean(active),
    networkMode: "always",
    retry: false,
  });

  return (
    <>
      <section className={styles.panel} aria-labelledby="runs-title">
        <div className={styles.toolbar}>
          <h2 id="runs-title">Завершённые запуски</h2>
          {!runs.isPending && (
            <button type="button" onClick={() => void runs.refetch()} disabled={runs.isFetching}>
              Обновить список
            </button>
          )}
        </div>
        {runs.isPending && <p role="status">Загружаем запуски…</p>}
        {runs.isError && <p className={styles.error} role="alert">{runs.error.message}</p>}
        {!runs.isError && runs.data?.length === 0 && (
          <p>Запусков пока нет. Выполните пакетное распознавание и обновите список.</p>
        )}
        {!runs.isError && active && (
          <>
            <div className={styles.picker}>
              <label htmlFor="run-picker">Запуск · дата завершения в UTC</label>
              <select
                id="run-picker"
                value={active.run_id}
                onChange={(event) => onRunChange(event.target.value)}
              >
                {runs.data?.map((run) => (
                  <option key={run.run_id} value={run.run_id}>
                    {utcDate.format(new Date(run.completed_at))} · N = {run.n} · {integer.format(run.flow_count)} потоков
                  </option>
                ))}
              </select>
            </div>
            <p className={styles.identifier}>Модель <code>{active.model_id}</code></p>
          </>
        )}
      </section>
      {!runs.isError && active && (
        <section className={styles.panel} aria-labelledby="stats-title">
          <div className={styles.toolbar}>
            <h2 id="stats-title">Сводка запуска</h2>
            {statistics.isError && (
              <button type="button" onClick={() => void statistics.refetch()} disabled={statistics.isFetching}>
                Повторить
              </button>
            )}
          </div>
          {statistics.isPending && <p role="status">Загружаем сводку…</p>}
          {statistics.isError && <p className={styles.error} role="alert">{statistics.error.message}</p>}
          {!statistics.isError && statistics.data && (
            <RunSummary statistics={statistics.data} />
          )}
        </section>
      )}
    </>
  );
}
