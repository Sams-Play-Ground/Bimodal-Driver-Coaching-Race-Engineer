"""
template_library.py — Default plain-language coaching templates, one or
more per detectable event, per proposal §3.5.5.

This is the coaching voice of the system — edit these before any real
deployment. Templates support {lap}, {turn}, and {prob} interpolation.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 6: Feedback generator).
"""

import json

DEFAULT_TEMPLATES = {
    "Off-Track": [
        "Lap {lap}: You ran wide at turn {turn}. Try to keep two wheels within the white line — track limits cost lap time plus momentum and in real conditions mean grass or barriers.",
        "Lap {lap}: Track limit violation detected. Focus on your reference points — pick a braking marker and a turn-in point earlier.",
    ],
    "Apex Miss": [
        "Lap {lap}: You missed the apex at turn {turn}. Late apex technique gives you earlier throttle application — try turning in slightly later.",
        "Lap {lap}: Apex deviation detected. A missed apex means you are carrying too much speed into the corner — trust the braking zone.",
    ],
    "Sliding": [
        "Lap {lap}: The car slid at turn {turn}. This indicates you are at or beyond the limit of grip — ease throttle application rate on exit.",
        "Lap {lap}: Instability detected. Check your brake release — a sudden release can upset the balance and provoke a slide.",
    ],
    "Brake Locked": [
        "Lap {lap}: Wheel lock detected. Try releasing the brake progressively over 20-30 metres rather than a sharp release to maintain tyre contact.",
        "Lap {lap}: Brake lock-up at turn {turn}. Your threshold braking pressure may be too high for this surface — modulate pressure earlier.",
    ],
    "Wheel Spin": [
        "Lap {lap}: Wheelspin detected on exit. Try rolling onto the throttle rather than snapping it open — especially in the lower gears.",
        "Lap {lap}: Excessive rear wheelspin. Smooth throttle application from the apex will give you faster acceleration than a sharp squirt.",
    ],
    "Trail Brake": [
        "Lap {lap}: Trail braking technique detected at turn {turn}. Ensure brake pressure is progressively decreasing as you unwind the steering.",
    ],
    "Aggressive Downshift": [
        "Lap {lap}: Aggressive downshift detected. The rev-match was too sharp — blip the throttle smoothly rather than spiking it.",
        "Lap {lap}: Downshift instability at turn {turn}. Try downshifting slightly earlier in the braking zone to give the drivetrain time to settle.",
    ],
    "Late Upshift": [
        "Lap {lap}: Late upshift detected. You stayed past the power peak — shift earlier to keep the engine in its power band.",
        "Lap {lap}: You are leaving revs on the table. Upshift when the power curve peaks, not when you hit the rev limiter.",
    ],
    "Rough Steering": [
        "Lap {lap}: Abrupt steering input detected at turn {turn}. Smooth, progressive steering reduces load transfer and keeps tyre grip consistent.",
        "Lap {lap}: Steering rate spike. Try to pre-plan your turn-in arc — late corrections are a sign the braking zone needs work.",
    ],
    "Gentle Accel": [
        "Lap {lap}: Good throttle application detected on exit. Smooth progressive power delivery like this protects rear grip effectively.",
    ],
}


def load_templates(json_path: str = None) -> dict:
    """Load a custom template set from JSON, falling back to DEFAULT_TEMPLATES."""
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    return DEFAULT_TEMPLATES


def save_templates(templates: dict, json_path: str):
    """Persist a (possibly edited) template set to JSON."""
    with open(json_path, "w") as f:
        json.dump(templates, f, indent=2)
