import { act, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { listFrom, optional, useSubmit } from "./Form";

describe("useSubmit", () => {
  it("runs one request at a time, even when asked twice in the same moment", async () => {
    let finish: () => void = () => {};
    const action = vi.fn(() => new Promise<string>((resolve) => (finish = () => resolve("done"))));
    const { result } = renderHook(() => useSubmit());

    let first: Promise<string | undefined> = Promise.resolve(undefined);
    let second: Promise<string | undefined> = Promise.resolve(undefined);
    act(() => {
      // Both before React re-renders: `pending` is still false for the second.
      first = result.current.run(action);
      second = result.current.run(action);
    });
    await act(async () => finish());

    expect(action).toHaveBeenCalledOnce();
    await expect(first).resolves.toBe("done");
    await expect(second).resolves.toBeUndefined();
    expect(result.current.pending).toBe(false);
  });

  it("keeps the API's error and clears it on the next attempt", async () => {
    const { result } = renderHook(() => useSubmit());

    await act(async () => {
      await result.current.run(() => Promise.reject(new ApiError(409, "Already applied")));
    });
    expect(result.current.error?.message).toBe("Already applied");

    await act(async () => {
      await result.current.run(() => Promise.resolve(1));
    });
    expect(result.current.error).toBeNull();
  });
});

describe("form values", () => {
  it("splits a typed list at commas and line breaks and drops blanks", () => {
    expect(listFrom(" Python, SQL ,,\n dbt \n")).toEqual(["Python", "SQL", "dbt"]);
    expect(listFrom("   ")).toEqual([]);
  });

  it("leaves a blank optional field out", () => {
    expect(optional("  ")).toBeUndefined();
    expect(optional(" Toronto ")).toBe("Toronto");
  });
});
