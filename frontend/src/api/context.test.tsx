import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { fakeApi, makeJob } from "../test/fakeApi";
import type { Job } from "./client";
import { ApiProvider, type Loadable, useLoad } from "./context";

describe("useLoad", () => {
  it("never hands a new key the previous key's data, and ignores a superseded answer", async () => {
    let releaseFirst: () => void = () => {};
    const api = fakeApi({
      // Job 1 answers only when released, after job 2 has already been asked for.
      "GET /jobs/1": () =>
        new Promise((resolve) => {
          releaseFirst = () => resolve({ body: makeJob({ id: 1, title: "First" }) });
        }),
      "GET /jobs/2": () => ({ body: makeJob({ id: 2, title: "Second" }) }),
      "GET /jobs/3": () => ({ body: makeJob({ id: 3, title: "Third" }) }),
    });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ApiProvider client={api.client}>{children}</ApiProvider>
    );
    const seen: { id: number; value: Loadable<Job> }[] = [];

    const { rerender } = renderHook(
      ({ id }: { id: number }) => {
        const [value] = useLoad(`job:${id}`, (client, signal) => client.getJob(id, { signal }));
        seen.push({ id, value });
        return value;
      },
      { wrapper, initialProps: { id: 3 } },
    );
    await waitFor(() => expect(seen.at(-1)?.value.state).toBe("ok"));

    rerender({ id: 1 });
    rerender({ id: 2 });
    await waitFor(() => expect(seen.at(-1)?.value.state).toBe("ok"));
    await act(async () => releaseFirst());

    for (const { id, value } of seen) {
      if (value.state === "ok") expect(value.data.id).toBe(id);
    }
    const last = seen.at(-1)!.value;
    expect(last.state === "ok" && last.data.title).toBe("Second");
  });

  it("loads again under the same key on reload", async () => {
    let count = 0;
    const api = fakeApi({ "GET /jobs/1": () => ({ body: makeJob({ title: `Load ${++count}` }) }) });
    const wrapper = ({ children }: { children: ReactNode }) => (
      <ApiProvider client={api.client}>{children}</ApiProvider>
    );

    const { result } = renderHook(
      () => useLoad("job:1", (client, signal) => client.getJob(1, { signal })),
      { wrapper },
    );
    await waitFor(() => expect(result.current[0].state).toBe("ok"));

    act(() => result.current[1]());

    await waitFor(() => {
      const value = result.current[0];
      expect(value.state === "ok" && value.data.title).toBe("Load 2");
    });
  });
});
