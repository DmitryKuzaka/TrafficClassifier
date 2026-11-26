import type { components } from "./schema";

export type PacketSequence = components["schemas"]["PredictRequest"];
export type Prediction = components["schemas"]["Prediction"];
type ReadyResponse = components["schemas"]["ReadyResponse"];

const unreachable = "Не удалось связаться с FastAPI. Проверьте, запущен ли сервер.";

async function requestJson<Result>(path: string, options?: RequestInit): Promise<Result> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (options?.signal?.aborted) {
      throw error;
    }
    throw new Error(unreachable, { cause: error });
  }

  if (response.status === 422) {
    throw new Error("API отклонил запрос. Проверьте время, размеры и направления пакетов.");
  }
  if (response.status === 503) {
    throw new Error("Модель недоступна. Проверьте сохранённые веса и перезапустите FastAPI.");
  }
  if (!response.ok) {
    throw new Error(`${unreachable} Код ответа: ${response.status}.`);
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

export function predictTraffic(packets: PacketSequence, signal?: AbortSignal) {
  return requestJson<Prediction>("/api/v1/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(packets),
    signal,
  });
}
