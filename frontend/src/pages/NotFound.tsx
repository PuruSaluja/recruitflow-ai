import { Link } from "react-router";

export function NotFound({ what }: { what?: string }) {
  return (
    <section>
      <h1>{what ? `No such ${what}` : "Page not found"}</h1>
      <p>
        <Link to="/jobs">Back to jobs</Link>
      </p>
    </section>
  );
}
