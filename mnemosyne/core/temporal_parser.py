"""
Natural Language Temporal Parser for Mnemosyne
===============================================
Extracts temporal references from natural language text and resolves
them to concrete dates. Supports:

- Absolute dates: "May 20, 2026", "2026-05-20", "20/05/2026"
- Relative dates: "today", "yesterday", "tomorrow"
- Day references: "last Monday", "this Monday", "next Monday"
- Week/month/year: "this week", "last month", "next year"
- Intervals: "2 days ago", "in 3 weeks"
- Named times: "morning", "afternoon", "evening"
- Vague: "recently", "lately", "a while ago"

Port of memory system's temporal-extraction.mjs to Python.

Usage:
    from mnemosyne.core.temporal_parser import extract_temporal, parse_nl_date
    
    result = extract_temporal("I met with Denis last Monday morning")
    # Returns: {
    #     "event_date": "2026-05-18",
    #     "event_date_precision": "day",
    #     "temporal_tags": ["monday", "2026-05-18", "week-21-2026", "morning"],
    # }
"""

import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict, List, Tuple


# Day name → weekday number (Monday=0, Sunday=6 per Python)
DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

NAMED_TIMES = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 6),
    "midnight": (0, 1),
    "noon": (12, 13),
    "dawn": (5, 7),
    "dusk": (18, 21),
}


def _resolve_relative_day(
    reference: datetime,
    day_name: str,
    qualifier: str = "this",  # this, last, next
) -> date:
    """
    Resolve a day name reference to an actual date.
    
    Args:
        reference: Reference datetime
        day_name: Day name (e.g., "monday")
        qualifier: "this", "last", or "next"
        
    Returns:
        Resolved date
    """
    target_wd = DAY_MAP.get(day_name.lower())
    if target_wd is None:
        return reference.date()
    
    current_wd = reference.weekday()
    
    if qualifier == "this":
        # This Monday = most recent past or current Monday if today
        diff = (current_wd - target_wd) % 7
        if diff == 0:
            return reference.date()  # Today is the target day
        return (reference - timedelta(days=diff)).date()
    
    elif qualifier == "last":
        # Last Monday = Monday of the PREVIOUS week
        # Formula: days_since_last = ((current_wd - target_wd + 7) % 7) + 7
        # From Monday: 0+7=7 days. From Tuesday: 1+7=8 days. From Sunday: 6+7=13 days.
        diff = ((current_wd - target_wd + 7) % 7) + 7
        return (reference - timedelta(days=diff)).date()
    
    elif qualifier == "next":
        # Next Monday = the upcoming Monday
        diff = (target_wd - current_wd) % 7
        if diff == 0:
            diff = 7  # "Next Monday" when today is Monday = 7 days from now
        return (reference + timedelta(days=diff)).date()
    
    return reference.date()


def parse_nl_date(
    text: str,
    reference: datetime = None,
) -> Optional[Tuple[date, str, List[str]]]:
    """
    Parse a natural language date expression and return (date, precision, tags).
    
    Args:
        text: Natural language text to parse
        reference: Reference datetime (default: now)
        
    Returns:
        (event_date, precision, temporal_tags) or None if no date found
    """
    if reference is None:
        reference = datetime.now()
    
    text_lower = text.lower().strip()
    
    # ---- Absolute dates ----
    # ISO format: 2026-05-20
    m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', text)
    if m:
        try:
            d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            d = None  # Invalid date like 2026-02-29
        if d is not None:
            week_num = d.isocalendar()[1]
            return (
                d,
                "day",
                [d.strftime("%Y-%m-%d"), f"week-{week_num}-{d.year}", d.strftime("%A").lower()]
            )
    
    # US/EU format: 05/20/2026 or 20/05/2026
    m = re.search(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', text)
    if m:
        try:
            a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
            # Heuristic: if first number > 12, it's day/month/year (EU)
            if a > 12:
                d = date(y, b, a)
            else:
                d = date(y, a, b)
            week_num = d.isocalendar()[1]
            return (
                d,
                "day",
                [d.strftime("%Y-%m-%d"), f"week-{week_num}-{d.year}", d.strftime("%A").lower()]
            )
        except ValueError:
            pass  # Invalid date, fall through
    
    # Named month + day: "May 20, 2026" or "May 20"
    m = re.search(
        r'\b(january|february|march|april|may|june|july|august|'
        r'september|october|november|december|'
        r'jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\s+(\d{1,2})(?:st|nd|rd|th)?'
        r'(?:,?\s*(\d{4}))?\b',
        text_lower
    )
    if m:
        try:
            month = MONTH_MAP.get(m.group(1), 1)
            day_num = int(m.group(2))
            year = int(m.group(3)) if m.group(3) else reference.year
            d = date(year, month, day_num)
            week_num = d.isocalendar()[1]
            return (
                d,
                "day",
                [d.strftime("%Y-%m-%d"), f"week-{week_num}-{d.year}", d.strftime("%A").lower()]
            )
        except ValueError:
            pass  # Invalid date like Feb 30, fall through
    
    # ---- Relative dates ----
    if re.search(r'\btoday\b', text_lower):
        d = reference.date()
        return (d, "day", [str(d), d.strftime("%A").lower()])
    
    if re.search(r'\byesterday\b', text_lower):
        d = (reference - timedelta(days=1)).date()
        return (d, "day", [str(d), d.strftime("%A").lower(), "yesterday"])
    
    if re.search(r'\btomorrow\b', text_lower):
        d = (reference + timedelta(days=1)).date()
        return (d, "day", [str(d), d.strftime("%A").lower(), "tomorrow"])
    
    if re.search(r'\bday before yesterday\b', text_lower) or re.search(r'\bday\s+before\s+yesterday\b', text_lower):
        d = (reference - timedelta(days=2)).date()
        return (d, "day", [str(d)])
    
    # ---- Day references: "last Monday", "this Monday", "next Monday" ----
    m = re.search(
        r'\b(last|this|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b',
        text_lower
    )
    if m:
        qualifier = m.group(1)
        day_name = m.group(2)
        d = _resolve_relative_day(reference, day_name, qualifier)
        week_num = d.isocalendar()[1]
        return (
            d,
            "day",
            [str(d), f"week-{week_num}-{d.year}", day_name, qualifier]
        )
    
    # Bare day name: "on Monday"
    m = re.search(
        r'\b(on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
        text_lower
    )
    if m:
        day_name = m.group(2)
        d = _resolve_relative_day(reference, day_name, "this")
        week_num = d.isocalendar()[1]
        return (
            d,
            "day",
            [str(d), f"week-{week_num}-{d.year}", day_name]
        )
    
    # ---- Week/Month/Year references ----
    # "this week", "last month", "next year"
    m = re.search(
        r'\b(this|last|next)\s+(week|month|year)\b',
        text_lower
    )
    if m:
        qualifier = m.group(1)
        unit = m.group(2)
        
        if qualifier == "this":
            if unit == "week":
                d = reference.date()
                return (d, "week", [f"week-{d.isocalendar()[1]}-{d.year}", "this-week"])
            elif unit == "month":
                d = reference.date()
                return (d, "month", [f"{d.year}-{d.month:02d}", "this-month"])
            elif unit == "year":
                d = reference.date()
                return (d, "year", [str(d.year), "this-year"])
        
        elif qualifier == "last":
            if unit == "week":
                d = (reference - timedelta(weeks=1)).date()
                return (d, "week", [f"week-{d.isocalendar()[1]}-{d.year}", "last-week"])
            elif unit == "month":
                # Last month
                if reference.month == 1:
                    d = date(reference.year - 1, 12, 1)
                else:
                    d = date(reference.year, reference.month - 1, 1)
                return (d, "month", [f"{d.year}-{d.month:02d}", "last-month"])
            elif unit == "year":
                d = date(reference.year - 1, 1, 1)
                return (d, "year", [str(d.year), "last-year"])
        
        elif qualifier == "next":
            if unit == "week":
                d = (reference + timedelta(weeks=1)).date()
                return (d, "week", [f"week-{d.isocalendar()[1]}-{d.year}", "next-week"])
            elif unit == "month":
                if reference.month == 12:
                    d = date(reference.year + 1, 1, 1)
                else:
                    d = date(reference.year, reference.month + 1, 1)
                return (d, "month", [f"{d.year}-{d.month:02d}", "next-month"])
            elif unit == "year":
                d = date(reference.year + 1, 1, 1)
                return (d, "year", [str(d.year), "next-year"])
    
    # ---- Interval: "2 days ago", "in 3 weeks", "3 hours ago" ----
    m = re.search(
        r'\b(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+(ago|before|earlier|back)\b',
        text_lower
    )
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        direction = m.group(3)
        
        delta_args = {}
        if unit == "second":
            delta_args["seconds"] = num
        elif unit == "minute":
            delta_args["minutes"] = num
        elif unit == "hour":
            delta_args["hours"] = num
        elif unit == "day":
            delta_args["days"] = num
        elif unit == "week":
            delta_args["weeks"] = num
        elif unit == "month":
            delta_args["days"] = num * 30
        elif unit == "year":
            delta_args["days"] = num * 365
        
        if direction in ("ago", "back", "before", "earlier"):
            try:
                d = (reference - timedelta(**delta_args)).date()
            except (OverflowError, ValueError):
                return None  # Extreme values like 999999 days
        else:
            try:
                d = (reference + timedelta(**delta_args)).date()
            except (OverflowError, ValueError):
                return None
        
        return (d, "day" if unit in ("day", "hour") else "week",
                [str(d), f"{num}-{unit}s-ago"])
    
    # "in 3 weeks" (future)
    m = re.search(
        r'\bin\s+(\d+)\s+(second|minute|hour|day|week|month|year)s?\b',
        text_lower
    )
    if m:
        num = int(m.group(1))
        unit = m.group(2)
        delta_args = {}
        if unit == "second": delta_args["seconds"] = num
        elif unit == "minute": delta_args["minutes"] = num
        elif unit == "hour": delta_args["hours"] = num
        elif unit == "day": delta_args["days"] = num
        elif unit == "week": delta_args["weeks"] = num
        elif unit == "month": delta_args["days"] = num * 30
        elif unit == "year": delta_args["days"] = num * 365
        
        try:
            d = (reference + timedelta(**delta_args)).date()
        except (OverflowError, ValueError):
            return None
        return (d, "day" if unit in ("day", "hour") else "week",
                [str(d), f"in-{num}-{unit}s"])
    
    # ---- Vague references ----
    if re.search(r'\b(recently|lately|not long ago)\b', text_lower):
        return (reference.date(), "relative", ["recently"])
    
    if re.search(r'\b(a while ago|some time ago|long ago)\b', text_lower):
        return (reference.date(), "relative", ["vague"])
    
    return None


def extract_temporal(
    text: str,
    reference: datetime = None,
) -> Dict:
    """
    Extract temporal information from text and return structured result.
    """
    result = parse_nl_date(text, reference)
    
    # Check for named time of day even when no date found
    tags = []
    text_lower = text.lower()
    for time_name in NAMED_TIMES:
        if time_name in text_lower:
            tags.append(time_name)
            break
    
    if result is None:
        return {
            "event_date": None,
            "event_date_precision": "unknown",
            "temporal_tags": tags,
            "primary_signal": tags[0] if tags else None,
        }
    
    event_date, precision, parsed_tags = result
    all_tags = parsed_tags + tags
    
    return {
        "event_date": event_date.isoformat() if event_date else None,
        "event_date_precision": precision,
        "temporal_tags": all_tags,
        "primary_signal": all_tags[0] if all_tags else None,
    }


def extract_date_from_text(
    text: str,
    reference: datetime = None,
) -> Optional[str]:
    """
    Convenience: extract just the ISO date string from text.
    
    Returns:
        "YYYY-MM-DD" or None
    """
    info = extract_temporal(text, reference)
    return info.get("event_date")
