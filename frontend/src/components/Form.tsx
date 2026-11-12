import { type ReactNode, useId, useRef, useState } from "react";

import { ApiError } from "../api/client";

/** The ids a field's control takes, so its label and hint are tied to it. */
export type FieldProps = { id: string; "aria-describedby"?: string };

/**
 * A labelled form control with an optional hint. The hint is linked to the
 * control by `aria-describedby` rather than put inside the label, so a screen
 * reader names the field by its label alone and reads the hint after it.
 */
export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: (props: FieldProps) => ReactNode;
}) {
  const id = useId();
  const hintId = `${id}-hint`;
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {/* One block under the label, so the fields on a line of a form row need two tracks. */}
      <div className="field-control">
        {children({ id, "aria-describedby": hint ? hintId : undefined })}
        {hint && (
          <p id={hintId} className="hint">
            {hint}
          </p>
        )}
      </div>
    </div>
  );
}

/** A list typed into one box: split at commas and line breaks, blanks dropped. */
export function listFrom(text: string): string[] {
  return text
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

/** An optional text field: its trimmed value, or undefined when left blank. */
export function optional(text: string): string | undefined {
  return text.trim() || undefined;
}

/**
 * The fields of `after` that differ from `before`: what an edit has to send.
 * Both are built the same way, one from the record as it was read and one
 * from the form, so a field nobody touched is never sent.
 */
export function changed<T extends object>(before: T, after: T): Partial<T> {
  const different: Partial<T> = {};
  for (const key of Object.keys(after) as (keyof T)[]) {
    if (JSON.stringify(before[key]) !== JSON.stringify(after[key])) different[key] = after[key];
  }
  return different;
}

/**
 * Runs a form's request once at a time and keeps its error. While it is
 * pending the form's button is disabled, so a double click sends one request.
 */
export function useSubmit() {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  // State lags a render behind; two clicks in one frame both see it false.
  const busy = useRef(false);

  async function run<T>(action: () => Promise<T>): Promise<T | undefined> {
    if (busy.current) return undefined;
    busy.current = true;
    setPending(true);
    setError(null);
    try {
      return await action();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught : new ApiError(0, "Something went wrong"));
      return undefined;
    } finally {
      busy.current = false;
      setPending(false);
    }
  }

  return { pending, error, run };
}
