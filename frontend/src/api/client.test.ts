import { describe, expect, it, vi } from "vitest";

import { ApiError, buildQuery, createClient } from "./client";

function respond(status: number, body: unknown, contentType = "application/json") {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return vi.fn(async () => new Response(text, { status, headers: { "Content-Type": contentType } }));
}

async function caught(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    if (error instanceof ApiError) return error;
    throw error;
  }
  throw new Error("expected the call to fail");
}

describe("buildQuery", () => {
  it("leaves out unset and empty values and encodes the rest", () => {
    expect(buildQuery({ q: "C++ & Go", status: undefined, department: "", limit: 20, offset: 0 }))
      .toBe("?q=C%2B%2B+%26+Go&limit=20&offset=0");
  });

  it("repeats a list's values, as the API's repeatable parameters expect", () => {
    expect(buildQuery({ id: [3, 1, 3], limit: 200 })).toBe("?id=3&id=1&id=3&limit=200");
    expect(buildQuery({ id: [] })).toBe("");
  });

  it("is empty when nothing is set", () => {
    expect(buildQuery()).toBe("");
    expect(buildQuery({ q: undefined })).toBe("");
  });
});

describe("createClient", () => {
  it("reads a job list from the API path with its filters", async () => {
    const page = { items: [], total: 0, limit: 20, offset: 0 };
    const fetchImpl = respond(200, page);
    const client = createClient(fetchImpl);

    await expect(client.listJobs({ status: "open", limit: 20 })).resolves.toEqual(page);

    expect(fetchImpl).toHaveBeenCalledOnce();
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/jobs?status=open&limit=20");
    expect(init.method).toBe("GET");
    expect(init.body).toBeUndefined();
  });

  it("sends a scoring request as JSON", async () => {
    const fetchImpl = respond(200, { job_id: 3, results: [] });
    const client = createClient(fetchImpl);

    await client.scoreJob(3, { candidate_ids: [1, 2], limit: 5 });

    const [url, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/v1/jobs/3/score");
    expect(init.method).toBe("POST");
    expect(init.body).toBe('{"candidate_ids":[1,2],"limit":5}');
    expect(new Headers(init.headers).get("Content-Type")).toBe("application/json");
  });

  it("turns the API's detail and code into an ApiError", async () => {
    const client = createClient(
      respond(503, { detail: "The trained matcher is missing", code: "matcher_unavailable" }),
    );

    const error = await caught(client.listMatches(7));

    expect(error.status).toBe(503);
    expect(error.code).toBe("matcher_unavailable");
    expect(error.message).toBe("The trained matcher is missing");
  });

  it("keeps a 404's detail, which carries no code", async () => {
    const error = await caught(createClient(respond(404, { detail: "Job 9 not found" })).getJob(9));

    expect(error.status).toBe(404);
    expect(error.code).toBeNull();
    expect(error.message).toBe("Job 9 not found");
  });

  it("summarises a 422's field errors by where they were found", async () => {
    const body = {
      detail: [
        { type: "less_than_equal", loc: ["query", "limit"], msg: "Input should be less than or equal to 100" },
        { type: "missing", loc: ["body", "title"], msg: "Field required" },
        { type: "value_error", msg: "salary_max must be greater" },
      ],
    };
    const error = await caught(createClient(respond(422, body)).listJobs({ limit: 1000 }));

    expect(error.status).toBe(422);
    expect(error.message).toBe(
      "query.limit: Input should be less than or equal to 100; title: Field required; salary_max must be greater",
    );
  });

  it("falls back to the status when an error body is not the API's", async () => {
    const error = await caught(
      createClient(respond(500, "<html>Server Error</html>", "text/html")).health(),
    );

    expect(error.status).toBe(500);
    expect(error.code).toBeNull();
    expect(error.message).toBe("Request failed (500)");
  });

  it("says a gateway error without the API's body means the API is down", async () => {
    const error = await caught(createClient(respond(502, "", "text/plain")).health());

    expect(error.status).toBe(502);
    expect(error.message).toBe("The API could not be reached (502)");
  });

  it("keeps the API's own 503 detail rather than the gateway wording", async () => {
    const error = await caught(
      createClient(respond(503, { detail: "Database unavailable", code: "db_unavailable" })).health(),
    );

    expect(error.message).toBe("Database unavailable");
    expect(error.code).toBe("db_unavailable");
  });

  it("refuses a successful answer that is not JSON", async () => {
    const error = await caught(createClient(respond(200, "<!doctype html>", "text/html")).health());

    expect(error.status).toBe(200);
    expect(error.message).toMatch(/other than JSON/);
  });

  it("reports an unreachable API as status 0", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    const error = await caught(createClient(fetchImpl).health());

    expect(error.status).toBe(0);
    expect(error.code).toBe("network_error");
  });

  it("lets an abort through as an abort", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    });

    await expect(createClient(fetchImpl).health()).rejects.toMatchObject({ name: "AbortError" });
  });

  it("passes the caller's signal to fetch", async () => {
    const fetchImpl = respond(200, { status: "ok" });
    const controller = new AbortController();

    await createClient(fetchImpl).health({ signal: controller.signal });

    const [, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(init.signal).toBe(controller.signal);
  });

  it("builds the report links under the API base", () => {
    const client = createClient(respond(200, {}), "/x");

    expect(client.shortlistPdfUrl(4)).toBe("/x/jobs/4/report.pdf");
    expect(client.shortlistPdfUrl(4, 15)).toBe("/x/jobs/4/report.pdf?limit=15");
    expect(client.scorecardPdfUrl(4, 12)).toBe("/x/jobs/4/candidates/12/scorecard.pdf");
  });

  it("sends each create as JSON to its collection", async () => {
    const fetchImpl = respond(201, { id: 1 });
    const client = createClient(fetchImpl);

    await client.createJob({ title: "Data Engineer", description: "Run the nightly loads." });
    await client.createCandidate({ full_name: "Ana Diaz", email: "ana@example.com", resume_text: "x".repeat(20) });
    await client.createApplication({ job_id: 3, candidate_id: 4, source: "referral" });

    const sent = (fetchImpl.mock.calls as unknown as [string, RequestInit][]).map(([url, init]) => [
      url,
      init.method,
      init.body,
      new Headers(init.headers).get("Content-Type"),
    ]);
    expect(sent).toEqual([
      ["/api/v1/jobs", "POST", '{"title":"Data Engineer","description":"Run the nightly loads."}', "application/json"],
      [
        "/api/v1/candidates",
        "POST",
        `{"full_name":"Ana Diaz","email":"ana@example.com","resume_text":"${"x".repeat(20)}"}`,
        "application/json",
      ],
      ["/api/v1/applications", "POST", '{"job_id":3,"candidate_id":4,"source":"referral"}', "application/json"],
    ]);
  });

  it("moves and reopens an application", async () => {
    const fetchImpl = respond(200, { id: 8 });
    const client = createClient(fetchImpl);

    await client.updateApplication(8, { status: "screening" });
    await client.reopenApplication(8);

    const sent = (fetchImpl.mock.calls as unknown as [string, RequestInit][]).map(([url, init]) => [
      url,
      init.method,
      init.body,
    ]);
    expect(sent).toEqual([
      ["/api/v1/applications/8", "PATCH", '{"status":"screening"}'],
      ["/api/v1/applications/8/reopen", "POST", "{}"],
    ]);
  });

  it("sends a change as a PATCH of the fields it names, a null among them", async () => {
    const fetchImpl = respond(200, { id: 5 });
    const client = createClient(fetchImpl);

    await client.updateJob(5, { title: "Data Engineer II", salary_max: null });
    await client.updateCandidate(9, { phone: null, skills: ["SQL"] });

    const sent = (fetchImpl.mock.calls as unknown as [string, RequestInit][]).map(([url, init]) => [
      url,
      init.method,
      init.body,
      new Headers(init.headers).get("Content-Type"),
    ]);
    expect(sent).toEqual([
      ["/api/v1/jobs/5", "PATCH", '{"title":"Data Engineer II","salary_max":null}', "application/json"],
      ["/api/v1/candidates/9", "PATCH", '{"phone":null,"skills":["SQL"]}', "application/json"],
    ]);
  });

  it("closes a job with a POST that carries no body", async () => {
    const fetchImpl = respond(200, { id: 5, status: "closed" });
    const client = createClient(fetchImpl);

    const job = await client.closeJob(5);

    expect(job.status).toBe("closed");
    const [[url, init]] = fetchImpl.mock.calls as unknown as [[string, RequestInit]];
    expect([url, init.method, init.body]).toEqual(["/api/v1/jobs/5/close", "POST", undefined]);
    expect(new Headers(init.headers).get("Content-Type")).toBeNull();
  });

  it("sends a resume as a multipart file and leaves its Content-Type to the browser", async () => {
    const fetchImpl = respond(200, { id: 5 });
    const client = createClient(fetchImpl);
    const resume = new File(["Python and SQL"], "cv.txt", { type: "text/plain" });

    await client.previewResume(resume);
    await client.uploadResume(5, resume);

    const calls = fetchImpl.mock.calls as unknown as [string, RequestInit][];
    expect(calls.map(([url, init]) => [url, init.method])).toEqual([
      ["/api/v1/candidates/resume-preview", "POST"],
      ["/api/v1/candidates/5/resume", "PUT"],
    ]);
    for (const [, init] of calls) {
      expect(init.body).toBeInstanceOf(FormData);
      expect((init.body as FormData).get("file")).toBe(resume);
      // Set by hand it would lack the boundary, and the API could not read the parts.
      expect(new Headers(init.headers).get("Content-Type")).toBeNull();
    }
  });

  it("reads a refused create's field errors into one message", async () => {
    const client = createClient(
      respond(422, {
        detail: [
          { loc: ["body", "title"], msg: "String should have at least 2 characters" },
          { loc: ["body", "salary_max"], msg: "Input should be greater than or equal to 0" },
        ],
      }),
    );

    const error = await caught(client.createJob({ title: " ", description: "x".repeat(20) }));

    expect(error.status).toBe(422);
    expect(error.message).toBe(
      "title: String should have at least 2 characters; salary_max: Input should be greater than or equal to 0",
    );
  });
});
