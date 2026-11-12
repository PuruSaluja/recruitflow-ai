/** The API's enum values the forms offer, in the order a person reads them. */

export const EMPLOYMENT_TYPES = ["full_time", "part_time", "contract", "internship"] as const;
export const SENIORITY_LEVELS = ["intern", "junior", "mid", "senior", "staff", "principal"] as const;
export const EDUCATION_LEVELS = ["high_school", "associate", "bachelor", "master", "doctorate"] as const;
/** Every status a job can hold. */
export const JOB_STATUSES = ["open", "paused", "draft", "closed"] as const;
