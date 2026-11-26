import { ModelStatus } from "./ModelStatus";
import { PacketEditor } from "./PacketEditor";

import styles from "./App.module.css";

export function App() {
  return (
    <>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <a className={styles.brand} href="#recognition">
            <span className={styles.brandMark} aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            Traffic<span>Classifier</span>
          </a>
          <nav className={styles.navigation} aria-label="Разделы">
            <a href="#recognition" aria-current="page">Распознавание</a>
            <button type="button" disabled>Аналитика</button>
          </nav>
          <ModelStatus />
        </div>
      </header>

      <main className={styles.main} id="recognition">
        <div className={styles.intro}>
          <h1>Распознавание трафика</h1>
          <p>Узнайте тип трафика по первым пакетам сетевого потока.</p>
        </div>

        <div className={styles.workspace}>
          <section className={styles.panel} aria-labelledby="packets-title">
            <PacketEditor />
          </section>

          <section className={styles.panel} aria-labelledby="prediction-title">
            <h2 id="prediction-title">Ответ модели</h2>
            <p className={styles.empty}>Здесь появится результат распознавания.</p>
          </section>
        </div>
      </main>
    </>
  );
}
