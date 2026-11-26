import { type FormEvent, useLayoutEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";

import {
  blankPacket,
  examplePackets,
  MAX_PACKETS,
  type PacketRow,
  type RowErrors,
  validatePackets,
} from "./packetRows";
import styles from "./PacketEditor.module.css";

type PacketFieldsProps = {
  index: number;
  row: PacketRow;
  errors?: RowErrors;
  onChange: (patch: Partial<PacketRow>) => void;
  onRemove: () => void;
};

function PacketFields({ index, row, errors, onChange, onRemove }: PacketFieldsProps) {
  const first = index === 0;
  const number = index + 1;
  const timeName = `time-${row.id}`;
  const sizeName = `size-${row.id}`;

  return (
    <div className={styles.row} role="group" aria-label={`Пакет ${number}`}>
      <span className={styles.number} aria-hidden="true">{number}</span>
      <div className={styles.field}>
        <label htmlFor={timeName} className={styles.label}>Время, мс</label>
        <input
          id={timeName}
          name={timeName}
          inputMode="decimal"
          autoComplete="off"
          readOnly={first}
          tabIndex={first ? -1 : undefined}
          value={row.timeMs}
          onChange={(event) => onChange({ timeMs: event.target.value })}
          aria-invalid={Boolean(errors?.timeMs)}
          aria-describedby={errors?.timeMs ? `${timeName}-error` : undefined}
        />
        {errors?.timeMs && <span className={styles.error} id={`${timeName}-error`}>{errors.timeMs}</span>}
      </div>
      <div className={styles.field}>
        <label htmlFor={sizeName} className={styles.label}>Размер, байт</label>
        <input
          id={sizeName}
          name={sizeName}
          inputMode="numeric"
          autoComplete="off"
          value={row.sizeBytes}
          onChange={(event) => onChange({ sizeBytes: event.target.value })}
          aria-invalid={Boolean(errors?.sizeBytes)}
          aria-describedby={errors?.sizeBytes ? `${sizeName}-error` : undefined}
        />
        {errors?.sizeBytes && <span className={styles.error} id={`${sizeName}-error`}>{errors.sizeBytes}</span>}
      </div>
      <div className={`${styles.field} ${styles.direction}`}>
        <label htmlFor={`direction-${row.id}`} className={styles.label}>Направление</label>
        <select
          id={`direction-${row.id}`}
          disabled={first}
          value={row.direction}
          onChange={(event) => onChange({ direction: event.target.value === "1" ? 1 : 0 })}
        >
          <option value="1">Как у первого пакета</option>
          <option value="0">В обратную сторону</option>
        </select>
      </div>
      {!first && (
        <button
          className={styles.remove}
          type="button"
          onClick={onRemove}
          aria-label={`Удалить пакет ${number}`}
        >
          <span aria-hidden="true">×</span>
        </button>
      )}
    </div>
  );
}

function focusField(form: HTMLFormElement | null, name: string) {
  const field = form?.elements.namedItem(name);
  if (field instanceof HTMLInputElement) {
    field.focus();
  }
}

export function PacketEditor() {
  const [rows, setRows] = useState<readonly PacketRow[]>(() => [blankPacket(0, true)]);
  const [submitted, setSubmitted] = useState(false);
  const form = useRef<HTMLFormElement>(null);
  const nextId = useRef(1);
  const pendingFocus = useRef<string | null>(null);
  const validation = validatePackets(rows);
  const errors = submitted && !validation.ok ? validation.errors : undefined;

  useLayoutEffect(() => {
    if (pendingFocus.current !== null) {
      focusField(form.current, pendingFocus.current);
      pendingFocus.current = null;
    }
  }, [rows]);

  function loadExample() {
    nextId.current = examplePackets.length;
    setSubmitted(false);
    setRows(examplePackets);
  }

  function addPacket() {
    if (rows.length >= MAX_PACKETS) {
      return;
    }
    const id = nextId.current++;
    pendingFocus.current = `time-${id}`;
    setRows((current) => [...current, blankPacket(id)]);
  }

  function updatePacket(id: number, patch: Partial<PacketRow>) {
    setRows((current) => current.map((row) => row.id === id ? { ...row, ...patch } : row));
  }

  function removePacket(index: number) {
    const target = rows[index + 1] ?? rows[index - 1];
    pendingFocus.current = `size-${target.id}`;
    setRows((current) => current.filter((_, position) => position !== index));
  }

  function clearPackets() {
    nextId.current = 1;
    pendingFocus.current = "size-0";
    setSubmitted(false);
    setRows([blankPacket(0, true)]);
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (validation.ok) {
      setSubmitted(true);
      return;
    }

    flushSync(() => setSubmitted(true));
    const row = rows.find(({ id }) => validation.errors[id]);
    if (row !== undefined) {
      const firstIssue = validation.errors[row.id].timeMs ? "time" : "size";
      focusField(form.current, `${firstIssue}-${row.id}`);
    }
  }

  return (
    <form className={styles.editor} ref={form} onSubmit={submit} noValidate>
      <div className={styles.heading}>
        <h2 id="packets-title">Пакеты потока</h2>
        <button className={styles.demo} type="button" onClick={loadExample}>
          Загрузить пример
        </button>
      </div>
      <p className={styles.hint}>Время каждого пакета отсчитывается от первого.</p>
      <div className={styles.table}>
        <div className={styles.columns} aria-hidden="true">
          <span>№</span>
          <span>Время, мс</span>
          <span>Размер, байт</span>
          <span>Направление</span>
        </div>
        {rows.map((row, index) => (
          <PacketFields
            key={row.id}
            index={index}
            row={row}
            errors={errors?.[row.id]}
            onChange={(patch) => updatePacket(row.id, patch)}
            onRemove={() => removePacket(index)}
          />
        ))}
      </div>

      <div className={styles.rowActions}>
        <button className={styles.secondary} type="button" onClick={addPacket} disabled={rows.length >= MAX_PACKETS}>
          <span aria-hidden="true">+</span> Добавить пакет
        </button>
        <span className={styles.count}>Пакетов: {rows.length} / {MAX_PACKETS}</span>
      </div>
      {errors && <p className={styles.feedback} role="alert">Исправьте отмеченные поля.</p>}
      {submitted && validation.ok && <p className={styles.feedback} role="status">Пакеты готовы к отправке.</p>}

      <div className={styles.footer}>
        <button className={styles.primary} type="submit">Проверить ввод</button>
        <button className={styles.secondary} type="button" onClick={clearPackets}>Очистить</button>
      </div>
    </form>
  );
}
