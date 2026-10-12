import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// Testing Library unmounts after each test by itself only when the runner
// exposes a global afterEach, which Vitest does not by default.
afterEach(cleanup);
