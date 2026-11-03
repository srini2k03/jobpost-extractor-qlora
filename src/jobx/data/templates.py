"""Template families: how a record's surface strings are laid out as a posting.

Families are the unit of the leakage-safe split: a family used for test never appears in
train. Each family has several variants plus random optional blocks, so posts within a
family still vary a lot.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from jobx import vocab
from jobx.data.surface import Surface

RESPONSIBILITIES: tuple[str, ...] = (
    "Own features end to end, from design to release",
    "Work closely with product and business teams",
    "Write clean, well-tested and maintainable code",
    "Participate in code reviews and design discussions",
    "Meet monthly targets and maintain daily reports",
    "Handle escalations and resolve issues on time",
    "Prepare weekly dashboards for leadership",
    "Mentor juniors and help them grow",
    "Coordinate with vendors and internal teams",
    "Improve processes and reduce turnaround time",
    "Track KPIs and suggest improvements",
    "Build relationships with key accounts",
    "Document processes and maintain records",
    "Drive quality and reliability across releases",
    "Support month-end closing activities",
    "Identify new opportunities in the assigned territory",
)
PERKS: tuple[str, ...] = (
    "Health insurance for self and family",
    "Flexible working hours",
    "Annual performance bonus",
    "Cab facility for late shifts",
    "Free lunch and snacks",
    "PF and gratuity as per norms",
    "Learning allowance and certifications",
    "Five-day work week",
    "ESOPs for high performers",
    "Paid parental leave",
    "Quarterly team outings",
    "Laptop provided",
)
INDUSTRIES: tuple[str, ...] = (
    "fintech",
    "e-commerce",
    "healthtech",
    "SaaS",
    "logistics",
    "edtech",
    "retail",
    "manufacturing",
    "insurance",
    "consumer internet",
    "BFSI",
    "automotive",
)
STREETS: tuple[str, ...] = (
    "MG Road",
    "Outer Ring Road",
    "Sector 62",
    "Hitech City Road",
    "Old Mahabalipuram Road",
    "Baner Road",
    "Salt Lake Sector V",
    "SG Highway",
    "Infopark Phase 2",
    "Cyber City",
)


@dataclass
class Ctx:
    rng: random.Random
    domain: str
    consultancy: str
    hq_city: str | None  # a city NOT in the gold locations (location-confusion trap), or None


def _b(rng: random.Random) -> str:
    return rng.choice(vocab.BULLETS)


def _skills_inline(s: Surface, rng: random.Random) -> str:
    sep = rng.choice([", ", ", ", " / ", " | ", ", "])
    return sep.join(s.skills)


def _resp(rng: random.Random, n: int) -> list[str]:
    return rng.sample(RESPONSIBILITIES, n)


def _about(s: Surface, ctx: Ctx) -> str | None:
    rng = ctx.rng
    ind = rng.choice(INDUSTRIES)
    if s.company is None:
        return rng.choice(
            [
                f"We are a fast-growing {ind} company.",
                f"Our client is a leading {ind} organisation.",
            ]
        )
    forms = [
        f"{s.company} is a fast-growing {ind} company serving customers across India.",
        f"About us: {s.company} builds products for the {ind} space.",
        f"{s.company} is one of the leading names in {ind}.",
    ]
    if ctx.hq_city:
        forms.append(f"{s.company} is a {ind} company headquartered in {ctx.hq_city}.")
    return rng.choice(forms)


def naukri_structured(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(3)
    labels = [
        {
            "title": "Job Title",
            "company": "Company",
            "exp": "Experience",
            "sal": "Salary",
            "loc": "Location",
            "skills": "Key Skills",
            "edu": "Education",
            "wm": "Work Mode",
            "et": "Employment Type",
            "sen": "Level",
        },
        {
            "title": "Role",
            "company": "Company Name",
            "exp": "Exp",
            "sal": "CTC",
            "loc": "Job Location",
            "skills": "Skills",
            "edu": "Qualification",
            "wm": "Work mode",
            "et": "Job Type",
            "sen": "Seniority",
        },
        {
            "title": "Designation",
            "company": "Organisation",
            "exp": "Experience Required",
            "sal": "Package",
            "loc": "Location(s)",
            "skills": "Must have",
            "edu": "Education",
            "wm": "Mode",
            "et": "Type",
            "sen": "Career Level",
        },
    ][v]
    lines = [f"{labels['title']}: {s.title}"]
    if s.company:
        lines.append(f"{labels['company']}: {s.company}")
    if s.seniority_line:
        lines.append(f"{labels['sen']}: {s.seniority_line}")
    if s.experience:
        lines.append(f"{labels['exp']}: {s.experience}")
    if s.salary:
        lines.append(f"{labels['sal']}: {s.salary}")
    elif rng.random() < 0.4:
        lines.append(
            f"{labels['sal']}: {rng.choice(['Not disclosed', 'As per industry standards', 'Best in industry'])}"
        )
    if s.locations:
        lines.append(f"{labels['loc']}: {s.locations}")
    if s.work_mode:
        lines.append(f"{labels['wm']}: {s.work_mode}")
    if s.employment_type:
        lines.append(f"{labels['et']}: {s.employment_type}")
    if s.education:
        lines.append(f"{labels['edu']}: {s.education}")
    if s.skills:
        lines.append(f"{labels['skills']}: {_skills_inline(s, rng)}")
    if rng.random() < 0.6:
        lines.append("")
        lines.append("Job Description:")
        lines += [f"{_b(rng)} {r}" for r in _resp(rng, rng.randint(2, 4))]
    return "\n".join(lines), v


def linkedin_prose(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(3)
    who = s.company or rng.choice(["We", "Our team", "A well-funded startup"])
    verb = "are" if who == "We" else "is"
    where = f" in {s.locations}" if s.locations else ""
    mode = f" ({s.work_mode})" if s.work_mode else ""
    paras = []
    if v == 0:
        paras.append(f"{who} {verb} hiring a {s.title}{where}{mode}!")
    elif v == 1:
        paras.append(
            f"Exciting opportunity: {s.title}{' at ' + s.company if s.company else ''}{where}{mode}."
        )
    else:
        paras.append(
            f"Join {s.company or 'us'} as a {s.title}.{(' Location: ' + s.locations + '.') if s.locations else ''}"
            f"{(' Work mode: ' + s.work_mode + '.') if s.work_mode else ''}"
        )
    about = _about(s, ctx)
    if about and rng.random() < 0.6:
        paras.append(about)
    req = []
    if s.experience and s.experience.lower().startswith("fresher"):
        req.append(rng.choice([f"{s.experience} can apply.", f"Experience: {s.experience}."]))
    elif s.experience:
        req.append(
            rng.choice(
                [
                    f"You bring {s.experience} of relevant experience.",
                    f"Experience: {s.experience}.",
                    f"We are looking for {s.experience} of experience.",
                ]
            )
        )
    if s.seniority_line:
        req.append(f"This is a {s.seniority_line} position.")
    if s.skills:
        req.append(
            rng.choice(
                [
                    f"Strong hands-on skills in {_skills_inline(s, rng)}.",
                    f"Tech/skills we care about: {_skills_inline(s, rng)}.",
                    f"You should be comfortable with {_skills_inline(s, rng)}.",
                ]
            )
        )
    if s.education:
        req.append(f"Education: {s.education}.")
    if req:
        paras.append(" ".join(req))
    tail = []
    if s.salary:
        tail.append(
            rng.choice(
                [f"Compensation: {s.salary}.", f"Salary range is {s.salary}.", f"CTC offered: {s.salary}."]
            )
        )
    if s.employment_type:
        tail.append(f"This is a {s.employment_type} role.")
    if tail:
        paras.append(" ".join(tail))
    if rng.random() < 0.5:
        paras.append("What you'll do: " + "; ".join(r.lower() for r in _resp(rng, 3)) + ".")
    paras.append(rng.choice(["Apply via the link below.", "DM me or apply directly.", "Referrals welcome!"]))
    return "\n\n".join(paras), v


def whatsapp_forward(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    e = lambda: rng.choice(vocab.EMOJIS)  # noqa: E731
    lines = [f"{e()}{e()} URGENT HIRING {e()}{e()}" if v == 0 else f"{e()} *Hiring Alert* {e()}"]
    lines.append(f"*{s.title.upper() if rng.random() < 0.4 else s.title}*")
    if s.company:
        lines.append(f"{e()} Company - {s.company}")
    if s.seniority_line:
        lines.append(f"{e()} Level - {s.seniority_line}")
    if s.experience:
        lines.append(f"{e()} Exp - {s.experience}")
    if s.salary:
        lines.append(f"{e()} Salary - {s.salary}")
    if s.locations:
        lines.append(f"📍 {s.locations}")
    if s.work_mode:
        lines.append(f"{e()} {s.work_mode}")
    if s.employment_type:
        lines.append(f"{e()} {s.employment_type}")
    if s.education:
        lines.append(f"{e()} Qualification - {s.education}")
    if s.skills:
        lines.append(f"{e()} Skills - {_skills_inline(s, rng)}")
    lines.append(rng.choice(vocab.HINGLISH_FRAGMENTS))
    lines.append(
        rng.choice(
            ["Share your CV on WhatsApp 👇", "Interested candidates DM", "Forward to needy friends 🙏"]
        )
    )
    return "\n".join(lines), v


def careers_page(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    sections: list[str] = []
    head = [s.title]
    meta = [x for x in (s.locations, s.work_mode, s.employment_type) if x]
    if meta:
        head.append(" · ".join(meta))
    sections.append("\n".join(head))
    about = _about(s, ctx)
    if about:
        sections.append(("About the company\n" if v == 0 else "Who we are\n") + about)
    sections.append("Responsibilities\n" + "\n".join(f"{_b(rng)} {r}" for r in _resp(rng, rng.randint(2, 5))))
    req = []
    if s.seniority_line:
        req.append(f"{s.seniority_line} level hire")
    if s.experience:
        req.append(f"{s.experience} of experience")
    req += s.skills
    if s.education:
        req.append(f"{s.education} or equivalent")
    if req:
        sections.append(
            ("Requirements\n" if v == 0 else "What we're looking for\n")
            + "\n".join(f"{_b(rng)} {r}" for r in req)
        )
    perks = rng.sample(PERKS, rng.randint(2, 4))
    if s.salary:
        perks.insert(0, f"Salary: {s.salary}")
    sections.append("Perks & Benefits\n" + "\n".join(f"{_b(rng)} {p}" for p in perks))
    return "\n\n".join(sections), v


def bullet_minimal(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    sep = " - " if v == 0 else ": "
    lines = [f"Role{sep}{s.title}"]
    if s.company:
        lines.append(f"Co{sep}{s.company}")
    if s.seniority_line:
        lines.append(f"Level{sep}{s.seniority_line}")
    if s.experience:
        lines.append(f"Exp{sep}{s.experience}")
    if s.salary:
        lines.append(f"CTC{sep}{s.salary}")
    if s.locations:
        lines.append(f"Loc{sep}{s.locations}")
    if s.work_mode:
        lines.append(f"Mode{sep}{s.work_mode}")
    if s.employment_type:
        lines.append(f"Type{sep}{s.employment_type}")
    if s.education:
        lines.append(f"Edu{sep}{s.education}")
    if s.skills:
        lines.append(f"Skills{sep}{_skills_inline(s, rng)}")
    return "\n".join(lines), v


def consultancy_mail(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    lines = ["Dear Candidate,", "", f"Greetings from {ctx.consultancy}!", ""]
    client = f"our client {s.company}" if s.company else "one of our reputed MNC clients"
    lines.append(f"We have an urgent opening for the position of {s.title} with {client}.")
    det = []
    if s.seniority_line:
        det.append(f"Level: {s.seniority_line}")
    if s.experience:
        det.append(f"Experience: {s.experience}")
    if s.salary:
        det.append(f"Budget: {s.salary}")
    if s.locations:
        det.append(f"Job Location: {s.locations}")
    if s.work_mode:
        det.append(f"Work Mode: {s.work_mode}")
    if s.employment_type:
        det.append(f"Employment: {s.employment_type}")
    if s.education:
        det.append(f"Qualification: {s.education}")
    if s.skills:
        det.append(f"Mandatory skills: {_skills_inline(s, rng)}")
    if det:
        lines.append("")
        lines.append("Job details:" if v == 0 else "Please find the details below:")
        lines += [f"{i + 1}. {d}" for i, d in enumerate(det)]
    lines += [
        "",
        "If interested, kindly share your updated resume with current CTC, expected CTC and notice period.",
        "",
        "Regards,",
        f"Recruitment Team, {ctx.consultancy}",
    ]
    return "\n".join(lines), v


def email_jd_hr(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    lines = [f"Subject: Opening for {s.title}" + (f" | {s.company}" if s.company else ""), ""]
    lines.append("Hi,")
    lines.append("")
    lines.append(f"Please find the JD for the {s.title} position below.")
    lines.append("")
    lines.append(f"Position: {s.title}")
    if s.seniority_line:
        lines.append(f"Grade/Level: {s.seniority_line}")
    if s.company:
        lines.append(f"Organisation: {s.company}")
    if s.locations:
        lines.append(f"Base location: {s.locations}")
    if s.work_mode:
        lines.append(f"Working model: {s.work_mode}")
    if s.experience:
        lines.append(f"Total experience: {s.experience}")
    if s.employment_type:
        lines.append(f"Nature of employment: {s.employment_type}")
    if s.salary:
        lines.append(f"Compensation: {s.salary}")
    lines.append(f"Reporting to: {rng.choice(['Head of Department', 'Team Manager', 'Director'])}")
    if s.skills:
        lines.append("")
        lines.append("Skill set:" if v == 0 else "Technical/functional skills:")
        lines += [f"  {_b(rng)} {x}" for x in s.skills]
    if s.education:
        lines.append(f"Education: {s.education}")
    lines += ["", "Thanks & Regards", "HR Team"]
    return "\n".join(lines), v


def walk_in_drive(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    d = rng.randint(1, 28)
    m = rng.randint(1, 12)
    who = s.company or "a leading company"
    lines = [f"WALK-IN DRIVE for {s.title} at {who}" if v == 0 else f"Walk-in Interview | {s.title}"]
    if v == 1 and s.company:
        lines.append(f"Company: {s.company}")
    lines.append(f"Date: {d:02d}/{m:02d}/2025, Time: 10 AM - 4 PM")
    if s.seniority_line:
        lines.append(f"Level: {s.seniority_line}")
    if s.experience:
        lines.append(f"Eligibility: {s.experience}")
    if s.education:
        lines.append(f"Qualification: {s.education}")
    if s.salary:
        lines.append(f"Salary: {s.salary}")
    if s.skills:
        lines.append(f"Required skills: {_skills_inline(s, rng)}")
    if s.locations:
        lines.append(f"Job location: {s.locations}")
    if s.work_mode:
        lines.append(f"Work mode: {s.work_mode}")
    if s.employment_type:
        lines.append(f"Job type: {s.employment_type}")
    lines.append(f"Venue: {rng.randint(2, 12)}th floor, {rng.choice(STREETS)}")
    lines.append(rng.choice(["Carry 2 copies of resume and a photo ID.", "Bring your updated CV."]))
    return "\n".join(lines), v


def tweet_short(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    parts = [("#hiring " if v == 0 else "We're hiring! ") + s.title]
    if s.company:
        parts.append(f"@ {s.company}")
    if s.seniority_line:
        parts.append(f"({s.seniority_line})")
    bits = [
        x for x in (s.experience, s.locations, s.work_mode, s.salary, s.employment_type, s.education) if x
    ]
    if s.skills:
        bits.append(_skills_inline(s, rng))
    text = " ".join(parts)
    if bits:
        text += " | " + " | ".join(bits)
    text += rng.choice([" DM for details", " Link in bio", " Apply 👇", ""])
    return text, v


def telegram_channel(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    lines = [rng.choice(["🔔 New Job Update", "📢 Off Campus / Experienced Hiring", "✅ Job Opportunity"])]
    if s.company:
        lines.append(f"Company: {s.company}")
    lines.append(f"Role: {s.title}")
    if s.seniority_line:
        lines.append(f"Level: {s.seniority_line}")
    if s.education:
        lines.append(f"Qualification: {s.education}")
    if s.experience:
        lines.append(f"Experience: {s.experience}")
    if s.salary:
        lines.append(f"Salary: {s.salary}")
    if s.locations:
        lines.append(f"Location: {s.locations}")
    if s.work_mode:
        lines.append(f"Work mode: {s.work_mode}")
    if s.employment_type:
        lines.append(f"Type: {s.employment_type}")
    if s.skills:
        lines.append(f"Skills: {_skills_inline(s, rng)}")
    lines.append(f"Apply Link: https://careers.example.in/jobs/{rng.randint(10000, 99999)}")
    if v == 1:
        lines.append("Join our channel for daily job updates 🔥")
    return "\n".join(lines), v


def hinglish_chat(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    lines = [rng.choice(["Hello sabko 👋", "Bhai log, ek opening hai", "Namaste dosto!"])]
    at = f" {s.company} mein" if s.company else " ek company mein"
    lines.append(f"{s.title} ki requirement hai{at}.")
    if s.seniority_line:
        lines.append(f"Level {s.seniority_line} hai.")
    if s.experience:
        lines.append(f"Experience chahiye {s.experience}.")
    if s.skills:
        lines.append(f"Skills: {_skills_inline(s, rng)} aana chahiye.")
    if s.salary:
        lines.append(rng.choice([f"Salary {s.salary} milegi.", f"Package {s.salary} tak hai."]))
    if s.locations:
        lines.append(f"Location {s.locations} hai.")
    if s.work_mode:
        lines.append(f"Mode: {s.work_mode}.")
    if s.employment_type:
        lines.append(f"Job type {s.employment_type} hai.")
    if s.education:
        lines.append(f"Qualification {s.education} honi chahiye.")
    lines.append(rng.choice(vocab.HINGLISH_FRAGMENTS))
    if v == 1:
        lines.append("Resume DM karo 🙏")
    return "\n".join(lines), v


def table_pipe(s: Surface, ctx: Ctx) -> tuple[str, int]:
    rng = ctx.rng
    v = rng.randrange(2)
    cols: list[tuple[str, str]] = [("Role", s.title)]
    if s.company:
        cols.append(("Company", s.company))
    if s.seniority_line:
        cols.append(("Level", s.seniority_line))
    if s.experience:
        cols.append(("Exp", s.experience))
    if s.salary:
        cols.append(("CTC", s.salary))
    if s.locations:
        cols.append(("Location", s.locations))
    if s.work_mode:
        cols.append(("Mode", s.work_mode))
    if s.employment_type:
        cols.append(("Type", s.employment_type))
    if s.education:
        cols.append(("Education", s.education))
    if s.skills:
        cols.append(("Skills", ", ".join(s.skills)))
    if v == 0:
        text = (
            "| "
            + " | ".join(c for c, _ in cols)
            + " |\n|"
            + "---|" * len(cols)
            + "\n| "
            + " | ".join(x for _, x in cols)
            + " |"
        )
    else:
        text = "\n".join(f"{c.ljust(10)}| {x}" for c, x in cols)
    return text, v


FAMILIES: dict[str, Callable[[Surface, Ctx], tuple[str, int]]] = {
    "naukri_structured": naukri_structured,
    "linkedin_prose": linkedin_prose,
    "whatsapp_forward": whatsapp_forward,
    "careers_page": careers_page,
    "bullet_minimal": bullet_minimal,
    "consultancy_mail": consultancy_mail,
    "email_jd_hr": email_jd_hr,
    "walk_in_drive": walk_in_drive,
    "tweet_short": tweet_short,
    "table_pipe": table_pipe,
    "telegram_channel": telegram_channel,
    "hinglish_chat": hinglish_chat,
}
# Families whose layout separates sections with blank lines (eligible for section shuffling).
SECTIONED = frozenset({"careers_page", "linkedin_prose", "naukri_structured"})
