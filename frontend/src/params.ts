/** Reading ids and offsets out of the address bar, where anyone can type anything. */

/** The API's largest id: every primary key is a PostgreSQL INTEGER. */
export const MAX_ID = 2_147_483_647;

/** A positive whole id within the API's range, or null for anything else. */
export function parseId(raw: string | undefined): number | null {
  if (!raw || !/^[1-9]\d*$/.test(raw)) return null;
  const id = Number(raw);
  return id <= MAX_ID ? id : null;
}

/** The `offset` parameter as a whole non-negative number; 0 when absent or unusable. */
export function offsetFrom(params: URLSearchParams): number {
  const raw = params.get("offset");
  if (!raw || !/^\d+$/.test(raw)) return 0;
  const value = Number(raw);
  return Number.isSafeInteger(value) ? value : 0;
}
