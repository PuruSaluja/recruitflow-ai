/** Where Previous goes: one page back, or from past the end to the start of the last page. */
export function previousOffset(total: number, limit: number, offset: number): number {
  if (offset >= total) return Math.max(0, Math.floor((total - 1) / limit) * limit);
  return Math.max(0, offset - limit);
}

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
  // An offset past the end comes from an old link or rows deleted since; say
  // so rather than numbering rows that are not there.
  const where =
    offset >= total ? `Past the last of ${total}` : `${offset + 1}–${Math.min(offset + limit, total)} of ${total}`;
  return (
    <nav className="pager" aria-label="Pages">
      <button
        type="button"
        disabled={offset === 0}
        onClick={() => onChange(previousOffset(total, limit, offset))}
      >
        Previous
      </button>
      <span className="muted">{where}</span>
      <button type="button" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)}>
        Next
      </button>
    </nav>
  );
}
