import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { api as defaultClient, ApiError, type ApiClient } from "./client";

const ApiContext = createContext<ApiClient>(defaultClient);

/** Supplies the client the pages call; tests pass one built on a fake `fetch`. */
export function ApiProvider({ client, children }: { client: ApiClient; children: ReactNode }) {
  return <ApiContext.Provider value={client}>{children}</ApiContext.Provider>;
}

export function useApi(): ApiClient {
  return useContext(ApiContext);
}

export type Loadable<T> =
  | { state: "loading" }
  | { state: "ok"; data: T }
  | { state: "error"; error: ApiError };

/**
 * Load something from the API whenever `key` changes.
 *
 * A change of `key` aborts the request in flight, so a slow answer for the
 * previous page can never overwrite the current one. `reload` fetches again
 * under the same key.
 */
export function useLoad<T>(
  key: string,
  load: (client: ApiClient, signal: AbortSignal) => Promise<T>,
): [Loadable<T>, () => void] {
  const client = useApi();
  const [result, setResult] = useState<{ key: string; value: Loadable<T> }>({
    key,
    value: { state: "loading" },
  });
  const [generation, setGeneration] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setResult({ key, value: { state: "loading" } });
    load(client, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) setResult({ key, value: { state: "ok", data } });
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        const apiError =
          error instanceof ApiError ? error : new ApiError(0, "Something went wrong reading the API");
        setResult({ key, value: { state: "error", error: apiError } });
      });
    return () => controller.abort();
    // `load` is a new function every render; `key` names everything it reads.
  }, [client, key, generation]);

  // Until the effect has run for a new key, the old key's result is stale.
  const value: Loadable<T> = result.key === key ? result.value : { state: "loading" };
  return [value, () => setGeneration((n) => n + 1)];
}
