import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { MODEL_READINESS_KEY, type PacketSequence, predictTraffic } from "./api/client";
import { ModelStatus } from "./ModelStatus";
import { PacketEditor } from "./PacketEditor";
import { PredictionPanel } from "./PredictionPanel";
import styles from "./App.module.css";

export function App() {
  const queryClient = useQueryClient();
  const requestController = useRef<AbortController | null>(null);
  const prediction = useMutation({
    mutationFn: ({ packets, signal }: { packets: PacketSequence; signal: AbortSignal }) =>
      predictTraffic(packets, signal),
    networkMode: "always",
    onSettled: async (_result, _error, { signal }) => {
      if (signal.aborted) {
        return;
      }
      await queryClient.cancelQueries({ queryKey: MODEL_READINESS_KEY });
      if (!signal.aborted) {
        void queryClient.invalidateQueries({ queryKey: MODEL_READINESS_KEY });
      }
    },
  });

  useEffect(() => () => requestController.current?.abort(), []);

  function requestPrediction(packets: PacketSequence) {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;
    prediction.mutate({ packets, signal: controller.signal });
  }

  function clearPrediction() {
    requestController.current?.abort();
    requestController.current = null;
    prediction.reset();
  }

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
            <PacketEditor
              onPredict={requestPrediction}
              onEdit={clearPrediction}
              isPending={prediction.isPending}
            />
          </section>

          <section className={styles.panel} aria-labelledby="prediction-title">
            <PredictionPanel
              result={prediction.data}
              error={prediction.error?.message}
              isPending={prediction.isPending}
            />
          </section>
        </div>
      </main>
    </>
  );
}
