import { TRAFFIC_CLASSES } from "../trafficClasses";
import type { components } from "./schema";

export type PacketSequence = components["schemas"]["PredictRequest"];
export type Prediction = components["schemas"]["Prediction"];
export type CompletedRun = components["schemas"]["CompletedRun"];
export type RunStatistics = components["schemas"]["PredictionStatistics"];
type ReadyResponse = components["schemas"]["ReadyResponse"];

export const MODEL_READINESS_KEY = ["model-readiness"] as const;

const unreachable = "Не удалось связаться с FastAPI. Проверьте, запущен ли сервер.";
const modelErrors: Record<number, string> = {
  422: "API отклонил запрос. Проверьте время, размеры и направления пакетов.",
  503: "Модель недоступна. Проверьте сохранённые веса и перезапустите FastAPI.",
};
const analyticsErrors: Record<number, string> = {
  404: "Выбранный запуск больше недоступен. Обновите список запусков.",
  503: "ClickHouse недоступен. Проверьте, запущена ли база, и повторите запрос.",
};

async function requestJson<Result>(
  path: string,
  options?: RequestInit,
  errors = modelErrors,
): Promise<Result> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (options?.signal?.aborted) {
      throw error;
    }
    throw new Error(unreachable, { cause: error });
  }

  if (!response.ok) {
    throw new Error(errors[response.status] ?? `${unreachable} Код ответа: ${response.status}.`);
  }

  try {
    return await response.json() as Result;
  } catch (error) {
    if (options?.signal?.aborted) {
      throw error;
    }
    throw new Error("API вернул ответ в неожиданном формате.", { cause: error });
  }
}

export async function getCompletedRuns(signal?: AbortSignal): Promise<CompletedRun[]> {
  const runs = await requestJson<CompletedRun[]>("/api/v1/runs", { signal }, analyticsErrors);
  if (!Array.isArray(runs) || !runs.every(isCompletedRun)) {
    throw new Error("API вернул список запусков в неожиданном формате.");
  }
  return runs;
}

function isCount(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

function isTimestamp(value: unknown): boolean {
  return typeof value === "string" && Number.isFinite(Date.parse(value));
}

function isCompletedRun(run: CompletedRun): boolean {
  return Boolean(
    run
    && typeof run.run_id === "string" && /^[0-9a-f]{64}$/.test(run.run_id)
    && typeof run.model_id === "string" && run.model_id.trim()
    && isCount(run.n) && run.n > 0
    && isCount(run.flow_count)
    && isTimestamp(run.completed_at),
  );
}

export async function getRunStatistics(runId: string, signal?: AbortSignal): Promise<RunStatistics> {
  const params = new URLSearchParams({ run_id: runId });
  const statistics = await requestJson<RunStatistics>(
    `/api/v1/statistics?${params}`,
    { signal },
    analyticsErrors,
  );
  if (statistics?.run_id !== runId) {
    throw new Error("API вернул сводку для другого запуска.");
  }
  const totals = statistics.totals;
  const labels = statistics.known_labels;
  const order = labels?.class_order;
  const matrix = labels?.confusion_matrix;
  if (
    !isCount(totals?.flows) || !isCount(totals.packets)
    || !isCount(totals.observed_packets) || !isCount(totals.bytes)
    || !TRAFFIC_CLASSES.every((category) =>
      isCount(statistics.by_predicted_category?.[category]?.flows))
    || statistics.bytes_basis !== "full_flow"
    || !isTimestamp(statistics.period_first_hour_utc)
    || !isTimestamp(statistics.period_last_hour_utc)
    || !Array.isArray(order) || order.length !== TRAFFIC_CLASSES.length
    || !TRAFFIC_CLASSES.every((category) => order.includes(category))
    || !Array.isArray(matrix) || matrix.length !== order.length
    || !matrix.every((row) => Array.isArray(row) && row.length === order.length && row.every(isCount))
    || !isCount(labels.correct)
    || !Number.isFinite(labels.accuracy) || labels.accuracy < 0 || labels.accuracy > 1
  ) {
    throw new Error("API вернул сводку запуска в неожиданном формате.");
  }
  return statistics;
}

export async function getModelReadiness(signal?: AbortSignal): Promise<ReadyResponse> {
  const ready = await requestJson<ReadyResponse>("/ready", { signal });
  if (
    ready?.status !== "ready"
    || typeof ready.model_id !== "string"
    || ready.model_id.trim() === ""
    || !Number.isInteger(ready.n)
    || ready.n < 1
  ) {
    throw new Error("На указанном адресе не найден API TrafficClassifier.");
  }
  return ready;
}

export async function predictTraffic(packets: PacketSequence, signal?: AbortSignal): Promise<Prediction> {
  const result = await requestJson<Prediction>("/api/v1/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(packets),
    signal,
  });

  const scoresAreValid = TRAFFIC_CLASSES.every((category) => {
    const score = result?.scores?.[category];
    return typeof score === "number" && Number.isFinite(score) && score >= 0 && score <= 1;
  });
  if (
    !scoresAreValid
    || !TRAFFIC_CLASSES.some((category) => category === result.category)
    || typeof result.model_id !== "string"
    || result.model_id.trim() === ""
    || !Number.isInteger(result.n)
    || !Number.isInteger(result.observed_packets)
    || result.observed_packets < 1
    || result.observed_packets > result.n
  ) {
    throw new Error("API вернул ответ модели в неожиданном формате.");
  }
  return result;
}
