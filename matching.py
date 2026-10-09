"""
PawCare - Lost & Found Structured Pet Matching Engine
Independent matching system based on structured attributes:
- Species (Primary gate & weight)
- Breed (Exact, partial, or cross-breed overlap)
- Gender (Exact match or unknown tolerance)
- Color & Markings (Color token intersection)
- Location & City (Geographic and neighborhood similarity)
- Date Proximity (Logical timeline: lost date precedes or coincides with found date)

Independent module: Does NOT depend on any disease ML models.
"""

from datetime import date, timedelta
import re


COMMON_COLORS = {
    "black", "white", "brown", "tan", "golden", "gold", "yellow", "red",
    "ginger", "orange", "grey", "gray", "silver", "cream", "fawn", "blue",
    "calico", "tortoiseshell", "tortie", "tabby", "brindle", "tricolor",
    "bicolor", "spotted", "sable", "chocolate", "caramel", "apricot"
}


def _extract_color_tokens(text):
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = set(cleaned.split())
    return words.intersection(COMMON_COLORS)


def _tokenize_text(text):
    if not text:
        return set()
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    # Filter out small stop words
    stop_words = {"the", "and", "near", "at", "in", "on", "by", "of", "with", "a", "an", "street", "st", "road", "rd", "ave", "avenue"}
    return {w for w in cleaned.split() if len(w) > 2 and w not in stop_words}


def calculate_match_score(report1, report2):
    """
    Computes a structured compatibility match score (0 - 100) between two Lost & Found reports.
    Typically one is a 'lost' report and one is a 'found' report.
    Returns:
        dict: {
            'score': int,
            'level': str,
            'badge_class': str,
            'reasons': list of dicts,
            'summary': str
        }
    """
    reasons = []
    total_score = 0

    # ---------------------------------------------------------
    # 1. Species Matching (30 pts - Strict Gateway)
    # ---------------------------------------------------------
    s1 = (report1.species or "").strip().lower()
    s2 = (report2.species or "").strip().lower()

    if not s1 or not s2 or s1 != s2:
        # Different species: immediate 0 score
        return {
            "score": 0,
            "level": "No Match",
            "badge_class": "bg-secondary",
            "reasons": [{"factor": "Species", "matched": False, "detail": f"Species mismatch ({report1.species} vs {report2.species})", "pts": 0}],
            "summary": "Different species cannot be matched."
        }

    total_score += 30
    reasons.append({
        "factor": "Species",
        "matched": True,
        "detail": f"Same species ({report1.species})",
        "pts": 30
    })

    # ---------------------------------------------------------
    # 2. Breed Matching (Up to 25 pts)
    # ---------------------------------------------------------
    b1 = (report1.breed or "").strip().lower()
    b2 = (report2.breed or "").strip().lower()

    if b1 and b2:
        if b1 == b2:
            total_score += 25
            reasons.append({
                "factor": "Breed",
                "matched": True,
                "detail": f"Exact breed match ({report1.breed})",
                "pts": 25
            })
        else:
            tokens1 = _tokenize_text(b1)
            tokens2 = _tokenize_text(b2)
            overlap = tokens1.intersection(tokens2)
            if overlap:
                total_score += 18
                reasons.append({
                    "factor": "Breed",
                    "matched": True,
                    "detail": f"Breed similarity ({report1.breed} ~ {report2.breed})",
                    "pts": 18
                })
            elif "mixed" in b1 or "mix" in b1 or "mixed" in b2 or "mix" in b2 or "unknown" in b1 or "unknown" in b2:
                total_score += 10
                reasons.append({
                    "factor": "Breed",
                    "matched": True,
                    "detail": "Mixed / unconfirmed breed candidate",
                    "pts": 10
                })
            else:
                reasons.append({
                    "factor": "Breed",
                    "matched": False,
                    "detail": f"Different breeds listed ({report1.breed} vs {report2.breed})",
                    "pts": 0
                })
    else:
        # One or both breeds unknown
        total_score += 8
        reasons.append({
            "factor": "Breed",
            "matched": True,
            "detail": "Breed unverified on one report",
            "pts": 8
        })

    # ---------------------------------------------------------
    # 3. Location & City Proximity (Up to 20 pts)
    # ---------------------------------------------------------
    c1 = (report1.city or "").strip().lower()
    c2 = (report2.city or "").strip().lower()
    loc1 = (report1.location or "").strip().lower()
    loc2 = (report2.location or "").strip().lower()

    loc_pts = 0
    loc_details = []

    if c1 and c2 and c1 == c2:
        loc_pts += 12
        loc_details.append(f"Same city/area ({report1.city})")
    elif c1 and c2:
        # Check partial city match (e.g. "North Seattle" vs "Seattle")
        if c1 in c2 or c2 in c1:
            loc_pts += 8
            loc_details.append(f"Nearby area ({report1.city} / {report2.city})")

    # Check neighborhood/street word overlap
    loc_tokens1 = _tokenize_text(loc1)
    loc_tokens2 = _tokenize_text(loc2)
    loc_overlap = loc_tokens1.intersection(loc_tokens2)

    if loc_overlap:
        shared_words = ", ".join(list(loc_overlap)[:2])
        loc_pts = min(20, loc_pts + 8)
        loc_details.append(f"Matching area keywords: '{shared_words}'")

    if loc_pts > 0:
        total_score += loc_pts
        reasons.append({
            "factor": "Location",
            "matched": True,
            "detail": " • ".join(loc_details),
            "pts": loc_pts
        })
    else:
        reasons.append({
            "factor": "Location",
            "matched": False,
            "detail": f"Different reported locations ({report1.city or report1.location} vs {report2.city or report2.location})",
            "pts": 0
        })

    # ---------------------------------------------------------
    # 4. Color & Appearance Matching (Up to 15 pts)
    # ---------------------------------------------------------
    colors1 = _extract_color_tokens(f"{report1.color or ''} {report1.description or ''}")
    colors2 = _extract_color_tokens(f"{report2.color or ''} {report2.description or ''}")
    color_overlap = colors1.intersection(colors2)

    if color_overlap:
        shared_colors = ", ".join(sorted(color_overlap))
        color_pts = min(15, 8 + (len(color_overlap) * 4))
        total_score += color_pts
        reasons.append({
            "factor": "Color / Coat",
            "matched": True,
            "detail": f"Shared color characteristics ({shared_colors})",
            "pts": color_pts
        })
    else:
        if report1.color and report2.color:
            reasons.append({
                "factor": "Color / Coat",
                "matched": False,
                "detail": f"Different colors noted ({report1.color} vs {report2.color})",
                "pts": 0
            })
        else:
            total_score += 5
            reasons.append({
                "factor": "Color / Coat",
                "matched": True,
                "detail": "Color details open / not fully specified",
                "pts": 5
            })

    # ---------------------------------------------------------
    # 5. Timeline / Date Proximity (Up to 10 pts)
    # ---------------------------------------------------------
    d1 = report1.report_date
    d2 = report2.report_date

    if d1 and d2:
        # Determine which is lost and which is found
        if report1.is_lost and report2.is_found:
            lost_date, found_date = d1, d2
        elif report1.is_found and report2.is_lost:
            lost_date, found_date = d2, d1
        else:
            lost_date, found_date = min(d1, d2), max(d1, d2)

        day_diff = (found_date - lost_date).days

        if 0 <= day_diff <= 3:
            total_score += 10
            reasons.append({
                "factor": "Date Proximity",
                "matched": True,
                "detail": f"Found within {day_diff} day{'s' if day_diff != 1 else ''} of being lost",
                "pts": 10
            })
        elif 4 <= day_diff <= 14:
            total_score += 8
            reasons.append({
                "factor": "Date Proximity",
                "matched": True,
                "detail": f"Found {day_diff} days after disappearing",
                "pts": 8
            })
        elif 15 <= day_diff <= 45:
            total_score += 5
            reasons.append({
                "factor": "Date Proximity",
                "matched": True,
                "detail": f"Reported within {day_diff} days",
                "pts": 5
            })
        elif day_diff < 0 and day_diff >= -2:
            # Tolerates slight reporting date offset (e.g. 1-2 days)
            total_score += 4
            reasons.append({
                "factor": "Date Proximity",
                "matched": True,
                "detail": "Dates reported within 48h window",
                "pts": 4
            })
        else:
            reasons.append({
                "factor": "Date Proximity",
                "matched": False,
                "detail": f"Dates separated by {abs(day_diff)} days",
                "pts": 0
            })

    # Final score capped to 100
    final_score = min(100, max(0, total_score))

    # Match Level classification
    if final_score >= 75:
        level = "High Potential Match"
        badge_class = "bg-success"
        summary = "Strong multi-attribute match across species, breed, location, and timeline."
    elif final_score >= 50:
        level = "Moderate Potential Match"
        badge_class = "bg-primary"
        summary = "Good correlation on key attributes; worth verifying photos and details."
    elif final_score >= 30:
        level = "Possible Match"
        badge_class = "bg-warning text-dark"
        summary = "Partial attribute overlap detected; review location and description."
    else:
        level = "Low Similarity"
        badge_class = "bg-secondary"
        summary = "Few matching attributes detected."

    return {
        "score": final_score,
        "level": level,
        "badge_class": badge_class,
        "reasons": reasons,
        "summary": summary
    }


def find_potential_matches(target_report, limit=8, min_score=30):
    """
    Finds potential matching reports for a given LostFoundReport.
    If target is 'lost', looks for 'found' reports in the same species.
    If target is 'found', looks for 'lost' reports in the same species.
    Returns a sorted list of matches:
    [
        {
            'report': LostFoundReport,
            'match': dict from calculate_match_score
        }
    ]
    """
    from models import LostFoundReport

    opposite_type = "found" if target_report.is_lost else "lost"

    # Query active candidate reports with opposite type and matching species
    candidates = LostFoundReport.query.filter(
        LostFoundReport.id != target_report.id,
        LostFoundReport.report_type == opposite_type,
        LostFoundReport.species.ilike(target_report.species),
        LostFoundReport.status == "active"
    ).all()

    scored_matches = []

    for candidate in candidates:
        match_info = calculate_match_score(target_report, candidate)
        if match_info["score"] >= min_score:
            scored_matches.append({
                "candidate": candidate,
                "report": candidate,
                "match": match_info
            })

    # Sort descending by score, then newest
    scored_matches.sort(
        key=lambda item: (item["match"]["score"], item["report"].created_at),
        reverse=True
    )

    return scored_matches[:limit]
