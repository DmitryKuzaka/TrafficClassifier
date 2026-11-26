import type { Prediction } from "./api/client";
import { classLabels, TRAFFIC_CLASSES } from "./trafficClasses";
import styles from "./PredictionPanel.module.css";

const percent = new Intl.NumberFormat("ru-RU", {
  style: "percent",
  maximumFractionDigits: 1,
});

type Props = {
  result?: Prediction;
  error?: string;
  isPending: boolean;
};

export function PredictionPanel({ result, error, isPending }: Props) {
  const category = TRAFFIC_CLASSES.find((item) => item === result?.category);

  return (
    <>
      <h2 id="prediction-title">Ответ модели</h2>
      <p className={styles.announcement} role="status" aria-atomic="true">
        {isPending ? "Модель обрабатывает пакеты…" : category ? `Предсказанный тип: ${classLabels[category]}` : ""}
      </p>
      {isPending && (
        <p className={styles.placeholder}>Модель обрабатывает пакеты…</p>
      )}
      {!isPending && error && <p className={styles.error} role="alert">{error}</p>}
      {!isPending && !error && !result && (
        <p className={styles.placeholder}>
          Загрузите пример или заполните таблицу, затем нажмите «Распознать».
        </p>
      )}
      {result && category && (
        <div className={styles.result}>
          <div className={styles.category} data-category={category}>
            <span>Предсказанный тип</span>
            <strong>{classLabels[category]}</strong>
          </div>
          <h3 className={styles.scoreHeading}>Оценки классов</h3>
          <ul className={styles.scores}>
            {TRAFFIC_CLASSES.map((item) => {
              const score = result.scores[item];
              return (
                <li key={item} data-category={item}>
                  <div className={styles.scoreLabel}>
                    <span>{classLabels[item]}</span>
                    <span className={styles.value}>{percent.format(score)}</span>
                  </div>
                  <div className={styles.track} aria-hidden="true">
                    <div className={styles.bar} style={{ width: `${score * 100}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
          <dl className={styles.details}>
            <div className={styles.packetCount}>
              <dt>В окне модели</dt>
              <dd>{result.observed_packets} из {result.n} пакетов</dd>
            </div>
            <div>
              <dt>Версия модели</dt>
              <dd><code className={styles.modelId}>{result.model_id}</code></dd>
            </div>
          </dl>
        </div>
      )}
    </>
  );
}
