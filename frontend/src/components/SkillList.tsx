/** Skills as tags, in the order the API lists them. */
// The generated read types leave defaulted lists optional, though the API always sends them.
export function SkillList({ skills = [] }: { skills?: string[] }) {
  if (skills.length === 0) return <span className="muted">None listed</span>;
  return (
    <ul className="tags">
      {skills.map((skill) => (
        <li key={skill} className="tag">
          {skill}
        </li>
      ))}
    </ul>
  );
}
