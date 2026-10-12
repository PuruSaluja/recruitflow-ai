/** Display helpers shared by the pages. */

/** `full_time` → "Full time". The API's enum values are snake_case words. */
export function label(value: string): string {
  const words = value.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const money = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

/**
 * A job's salary band. The API sends salaries as decimal strings, so they are
 * parsed here only to be grouped for reading.
 */
export function salaryBand(min: string | null | undefined, max: string | null | undefined): string {
  const show = (value: string) => money.format(Number(value));
  if (min && max) return min === max ? show(min) : `${show(min)} – ${show(max)}`;
  if (min) return `from ${show(min)}`;
  if (max) return `up to ${show(max)}`;
  return "Not stated";
}

const dateFormat = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric" });

export function date(iso: string): string {
  return dateFormat.format(new Date(iso));
}

/** A share in 0..1 as a whole percentage. */
export function percent(share: number): string {
  return `${Math.round(share * 100)}%`;
}

export function years(count: number): string {
  return count === 1 ? "1 year" : `${count} years`;
}
