import type { RunStatistics } from "./api/client";
import { classLabels, shortClassLabels, TRAFFIC_CLASSES } from "./trafficClasses";
import styles from "./RunSummary.module.css";

const integer = new Intl.NumberFormat("ru-RU");
const percent = new Intl.NumberFormat("ru-RU", { style: "percent", maximumFractionDigits: 1 });
const hour = new Intl.DateTimeFormat("ru-RU", {
  dateStyle: "short",
  timeStyle: "short",
  timeZone: "UTC",
});

export function RunSummary({ statistics }: { statistics: RunStatistics }) {
  const { totals, known_labels: labels } = statistics;
  const firstHour = hour.format(new Date(statistics.period_first_hour_utc));
  const lastHour = hour.format(new Date(statistics.period_last_hour_utc));
  const labeled = labels.confusion_matrix.flat().reduce((sum, count) => sum + count, 0);

  return (
    <>
      <p className={styles.period}>
        Часы начала потоков · {firstHour}{firstHour !== lastHour && ` — ${lastHour}`} UTC
      </p>
      <dl className={styles.metrics}>
        <div>
          <dt>Потоки</dt>
          <dd>{integer.format(totals.flows)}</dd>
        </div>
        <div>
          <dt>Все пакеты в записи</dt>
          <dd>{integer.format(totals.packets)}</dd>
        </div>
        <div>
          <dt>Пакеты, увиденные моделью</dt>
          <dd>{integer.format(totals.observed_packets)}</dd>
        </div>
        <div>
          <dt>Объём полных потоков</dt>
          <dd>{integer.format(totals.bytes)} <span>байт</span></dd>
        </div>
      </dl>

      <div className={styles.charts}>
        <section className={styles.chart} aria-labelledby="distribution-title">
          <h3 id="distribution-title">Распределение по прогнозам</h3>
          <ul className={styles.distribution}>
            {TRAFFIC_CLASSES.map((category) => {
              const count = statistics.by_predicted_category[category].flows;
              const share = totals.flows === 0 ? 0 : count / totals.flows;
              return (
                <li key={category} data-category={category}>
                  <div className={styles.barLabel}>
                    <span>{classLabels[category]}</span>
                    <span>{integer.format(count)} <span className={styles.share}>· {percent.format(share)}</span></span>
                  </div>
                  <div className={styles.track} aria-hidden="true">
                    <div className={styles.bar} style={{ width: `${Math.min(share, 1) * 100}%` }} />
                  </div>
                </li>
              );
            })}
          </ul>
        </section>

        <section className={styles.chart} aria-labelledby="matrix-title">
          <h3 id="matrix-title">Матрица ошибок</h3>
          {labeled === 0 ? <p>Для этого запуска нет известных меток.</p> : (
            <>
              <p className={styles.axis} id="matrix-axes">
                Строки — известный класс, столбцы — прогноз модели.
              </p>
              <table className={styles.matrix} aria-labelledby="matrix-title" aria-describedby="matrix-axes">
                <thead>
                  <tr>
                    <td />
                    {labels.class_order.map((category) => (
                      <th key={category} scope="col" aria-label={classLabels[category]}>
                        {shortClassLabels[category]}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {labels.class_order.map((category, row) => (
                    <tr key={category}>
                      <th scope="row" aria-label={classLabels[category]}>{shortClassLabels[category]}</th>
                      {labels.class_order.map((predicted, column) => (
                        <td key={predicted} data-correct={row === column}>
                          {integer.format(labels.confusion_matrix[row][column])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className={styles.accuracy}>
                Верные ответы в выбранном запуске{" "}
                <strong>{percent.format(labels.accuracy)}</strong>
              </p>
            </>
          )}
        </section>
      </div>
    </>
  );
}
