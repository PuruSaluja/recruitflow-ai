/** Previous/next over an offset-paginated listing; hidden when it fits on one page. */
export function Pager({
  total,
  limit,
  offset,
  onChange,
}: {
  total: number;
  limit: number;
  offset: number;
  onChange: (offset: number) => void;
}) {
  if (total <= limit && offset === 0) return null;
  const first = total === 0 ? 0 : Math.min(offset + 1, total);
  const last = Math.min(offset + limit, total);
  return (
    <nav className="pager" aria-label="Pages">
      <button type="button" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>
        Previous
      </button>
      <span className="muted">
        {first}–{last} of {total}
      </span>
      <button type="button" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)}>
        Next
      </button>
    </nav>
  );
}
