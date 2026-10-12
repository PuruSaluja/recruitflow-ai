import type { ReactNode } from "react";

import type { ApiError } from "../api/client";
import type { Loadable } from "../api/context";

export function Loading({ what }: { what: string }) {
  return (
    <p className="muted" role="status">
      Loading {what}…
    </p>
  );
}

export function ErrorMessage({ error, retry }: { error: ApiError; retry?: () => void }) {
  return (
    <div className="error-box" role="alert">
      <p>{error.message}</p>
      {retry && (
        <button type="button" onClick={retry}>
          Try again
        </button>
      )}
    </div>
  );
}

/** Loading text, the error, or `children` with the data. */
export function Loaded<T>({
  value,
  what,
  retry,
  children,
}: {
  value: Loadable<T>;
  what: string;
  retry?: () => void;
  children: (data: T) => ReactNode;
}) {
  if (value.state === "loading") return <Loading what={what} />;
  if (value.state === "error") return <ErrorMessage error={value.error} retry={retry} />;
  return <>{children(value.data)}</>;
}
