"""
persona_classifier.py — Driving Persona classification from a session's
event counts, per proposal §3.3.

A simple rule-based classifier (not a trained model): compares aggregate
counts of aggression-linked events against smoothness- and caution-linked
events to label the session's overall driving character.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 6: Feedback generator — Driving Persona classification).
"""

PERSONA_LABELS = ("Aggressive", "Smooth", "Cautious")


def classify_persona(event_counts: dict) -> str:
    """
    Classify overall driving style for a session from its event_counts dict
    (as produced by FeedbackGenerator.session_summary).

    Aggressive: high Brake Locked + Wheel Spin + Rough Steering counts.
    Smooth:     high Gentle Accel count.
    Cautious:   high Late Upshift count.

    Ties resolve toward Smooth (a conservative default coaching label).
    """
    aggressive_score = (event_counts.get("Brake Locked", 0)
                         + event_counts.get("Wheel Spin", 0)
                         + event_counts.get("Rough Steering", 0))
    smooth_score = event_counts.get("Gentle Accel", 0)
    cautious_score = event_counts.get("Late Upshift", 0)

    if aggressive_score > smooth_score and aggressive_score > cautious_score:
        return "Aggressive"
    elif smooth_score >= aggressive_score and smooth_score >= cautious_score:
        return "Smooth"
    else:
        return "Cautious"
