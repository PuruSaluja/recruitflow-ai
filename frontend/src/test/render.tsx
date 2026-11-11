import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router";

import type { ApiClient } from "../api/client";
import { ApiProvider } from "../api/context";
import { AppRoutes } from "../App";

export const HEALTH = {
  "GET /health": () => ({ body: { status: "ok", version: "0.1.0", database: "ok" } }),
};

/** The whole app at `path`, talking to `client`. */
export function renderAt(path: string, client: ApiClient) {
  return render(
    <ApiProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </ApiProvider>,
  );
}
