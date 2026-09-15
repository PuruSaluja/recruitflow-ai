"""Skills taxonomy: canonical skill names, the spellings that mean them, and their categories.

Profiles and job descriptions name one skill many ways ("Postgres", "PostgreSQL",
"postgresql"), so the matcher compares skills by canonical name. A spelling the
taxonomy does not know is kept as written, since an unknown skill is still evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

CATEGORIES = frozenset(
    {
        "analytics",
        "backend",
        "data",
        "database",
        "frontend",
        "infrastructure",
        "language",
        "ml",
        "practice",
        "testing",
    }
)


@dataclass(frozen=True)
class Skill:
    name: str
    category: str
    aliases: tuple[str, ...] = ()


SKILLS: tuple[Skill, ...] = (
    Skill("Python", "language", ("python3",)),
    Skill("SQL", "language"),
    Skill("R", "language", ("rstats",)),
    Skill("Java", "language"),
    Skill("Scala", "language"),
    Skill("Go", "language", ("golang",)),
    Skill("JavaScript", "language", ("js", "ecmascript")),
    Skill("TypeScript", "language", ("ts",)),
    Skill("C#", "language", ("csharp",)),
    Skill("C++", "language", ("cpp",)),
    Skill("Bash", "language", ("shell scripting",)),
    Skill("Pandas", "data"),
    Skill("NumPy", "data"),
    Skill("Apache Spark", "data", ("spark", "pyspark")),
    Skill("Apache Airflow", "data", ("airflow",)),
    Skill("dbt", "data", ("data build tool",)),
    Skill("Kafka", "data", ("apache kafka",)),
    Skill("ETL", "data", ("etl pipelines",)),
    Skill("Data Modeling", "data", ("data modelling", "dimensional modeling")),
    Skill("Statistics", "data", ("statistical analysis",)),
    Skill("A/B Testing", "data", ("ab testing", "experimentation")),
    Skill("Excel", "analytics", ("microsoft excel",)),
    Skill("Tableau", "analytics"),
    Skill("Power BI", "analytics", ("powerbi",)),
    Skill("Looker", "analytics"),
    Skill("Data Visualization", "analytics", ("data visualisation",)),
    Skill("Machine Learning", "ml", ("ml",)),
    Skill("scikit-learn", "ml", ("sklearn", "scikit learn")),
    Skill("TensorFlow", "ml"),
    Skill("PyTorch", "ml", ("torch",)),
    Skill("Deep Learning", "ml"),
    Skill("NLP", "ml", ("natural language processing",)),
    Skill("Computer Vision", "ml"),
    Skill("MLOps", "ml", ("ml ops",)),
    Skill("Feature Engineering", "ml"),
    Skill("XGBoost", "ml"),
    Skill("AWS", "infrastructure", ("amazon web services",)),
    Skill("GCP", "infrastructure", ("google cloud", "google cloud platform")),
    Skill("Azure", "infrastructure", ("microsoft azure",)),
    Skill("Docker", "infrastructure"),
    Skill("Kubernetes", "infrastructure", ("k8s",)),
    Skill("Terraform", "infrastructure"),
    Skill("CI/CD", "infrastructure", ("ci cd", "continuous integration")),
    Skill("Linux", "infrastructure"),
    Skill("Prometheus", "infrastructure"),
    Skill("Ansible", "infrastructure"),
    Skill("PostgreSQL", "database", ("postgres",)),
    Skill("MySQL", "database"),
    Skill("MongoDB", "database", ("mongo",)),
    Skill("Redis", "database"),
    Skill("Snowflake", "database"),
    Skill("BigQuery", "database"),
    Skill("FastAPI", "backend"),
    Skill("Django", "backend"),
    Skill("Flask", "backend"),
    Skill("Spring Boot", "backend", ("spring",)),
    Skill("Node.js", "backend", ("nodejs", "node")),
    Skill("REST APIs", "backend", ("rest", "rest api", "restful apis")),
    Skill("GraphQL", "backend"),
    Skill("Microservices", "backend"),
    Skill("React", "frontend", ("react.js", "reactjs")),
    Skill("Vue.js", "frontend", ("vue", "vuejs")),
    Skill("Angular", "frontend"),
    Skill("HTML", "frontend", ("html5",)),
    Skill("CSS", "frontend", ("css3",)),
    Skill("Redux", "frontend"),
    Skill("Accessibility", "frontend", ("a11y", "wcag")),
    Skill("pytest", "testing"),
    Skill("Selenium", "testing"),
    Skill("Cypress", "testing"),
    Skill("Jest", "testing"),
    Skill("Test Automation", "testing", ("automated testing",)),
    Skill("Git", "practice"),
    Skill("Agile", "practice", ("scrum",)),
    Skill("System Design", "practice", ("systems design",)),
)


def spelling_key(spelling: str) -> str:
    """How spellings are compared: whitespace collapsed and case folded."""
    return " ".join(spelling.split()).casefold()


def build_lookup(skills: Iterable[Skill]) -> dict[str, str]:
    """Map every spelling to its skill's canonical name.

    Raises ``ValueError`` when two skills claim one spelling, since the matcher
    could then not tell them apart.
    """
    lookup: dict[str, str] = {}
    for skill in skills:
        for spelling in (skill.name, *skill.aliases):
            owner = lookup.setdefault(spelling_key(spelling), skill.name)
            if owner != skill.name:
                raise ValueError(f"{spelling!r} is claimed by both {owner!r} and {skill.name!r}")
    return lookup


_LOOKUP = build_lookup(SKILLS)
_BY_NAME = {skill.name: skill for skill in SKILLS}


def get_skill(name: str) -> Skill:
    """The skill with this canonical name; ``KeyError`` for anything else, aliases included."""
    return _BY_NAME[name]


def canonical_skill(spelling: str) -> str | None:
    """The canonical name ``spelling`` refers to, or None if the taxonomy does not know it."""
    return _LOOKUP.get(spelling_key(spelling))


def normalise_skills(spellings: Iterable[str]) -> list[str]:
    """Canonicalise known spellings, keep unknown ones tidied, and drop blanks and repeats.

    First-seen order is kept, so ``["Postgres", "SQL", "postgresql"]`` becomes
    ``["PostgreSQL", "SQL"]``.
    """
    seen: set[str] = set()
    result: list[str] = []
    for spelling in spellings:
        tidied = " ".join(spelling.split())
        if not tidied:
            continue
        name = canonical_skill(tidied) or tidied
        key = spelling_key(name)
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result
