import type { PacketSequence } from "./api/client";

export const MAX_PACKETS = 64;

export type PacketRow = {
  id: number;
  timeMs: string;
  sizeBytes: string;
  direction: 0 | 1;
};

type Field = "timeMs" | "sizeBytes";
export type RowErrors = Partial<Record<Field, string>>;

type Validation =
  | { ok: true; packets: PacketSequence }
  | { ok: false; errors: Record<number, RowErrors> };

export function blankPacket(id: number, first = false): PacketRow {
  return {
    id,
    timeMs: first ? "0" : "",
    sizeBytes: "",
    direction: 1,
  };
}

export const examplePackets: readonly PacketRow[] = [
  { id: 0, timeMs: "0", sizeBytes: "100", direction: 1 },
  { id: 1, timeMs: "2", sizeBytes: "300", direction: 0 },
  { id: 2, timeMs: "5,250", sizeBytes: "180", direction: 1 },
];

function toMicroseconds(value: string): number | null {
  const text = value.trim();
  if (!/^(?:\d+|\d*[.,]\d{1,3})$/.test(text)) {
    return null;
  }

  const [whole, fraction = ""] = text.split(/[.,]/);
  const offset = Number(whole) * 1000 + Number(fraction.padEnd(3, "0"));
  return Number.isSafeInteger(offset) ? offset : null;
}

function toSize(value: string): number | null {
  const text = value.trim();
  const size = Number(text);
  return /^\d+$/.test(text) && size <= 2 ** 32 - 1 ? size : null;
}

export function validatePackets(rows: readonly PacketRow[]): Validation {
  const errors: Record<number, RowErrors> = {};
  const packets: PacketSequence = { offsets_us: [], sizes_bytes: [], directions: [] };
  let previousOffset = 0;

  for (const row of rows) {
    const time = toMicroseconds(row.timeMs);
    const size = toSize(row.sizeBytes);
    const issues: RowErrors = {};

    if (time === null) {
      issues.timeMs = "Укажите время в мс, до трёх знаков после запятой.";
    } else if (time < previousOffset) {
      issues.timeMs = "Время не может быть меньше предыдущего.";
    } else {
      previousOffset = time;
    }

    if (size === null) {
      issues.sizeBytes = "Укажите целый размер от 0 до 4 294 967 295 байт.";
    }

    if (time === null || size === null || issues.timeMs) {
      errors[row.id] = issues;
      continue;
    }

    packets.offsets_us.push(time);
    packets.sizes_bytes.push(size);
    packets.directions.push(row.direction);
  }

  return Object.keys(errors).length > 0 ? { ok: false, errors } : { ok: true, packets };
}
