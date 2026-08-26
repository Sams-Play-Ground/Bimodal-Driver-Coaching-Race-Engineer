"""
Coaching feedback generator — ported from cnn-module-codes.ipynb
Section 16.4, SECTION 6 (FeedbackGenerator). Turns fusion-head
probabilities into plain-language coaching lines, a per-session event
count, and a driving-persona label. This is what should populate the
"Driving coach" report preview and the saved PDF.

Note: the notebook's SHAP-confirmation gate is intentionally left
disabled here (shap_summary=None -> always True) — the app doesn't ship
a live SHAP explainer, so gating on it would silently suppress every
piece of feedback. If you want that gate active in the app, export
run_fusion_shap()'s per-class importances from the notebook to a JSON
and pass it in as shap_summary.
"""
from typing import Optional, List, Dict
import numpy as np

from core.fusion_engine import FUSED_CLASS_NAMES, FUSED_LABEL_COLS

# 42 default feedback templates — one or more per detectable event.
# Edit freely — this is the coaching voice of the app.
DEFAULT_TEMPLATES: Dict[str, List[str]] = {
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


class FeedbackGenerator:
    """Logic gate (proposal §3.5.5): an event fires a feedback item only
    if prob > threshold (SHAP confirmation optional, disabled by default
    in the app — see module docstring)."""

    def __init__(self, thresholds: List[float], templates: Optional[dict] = None,
                 shap_summary: Optional[dict] = None):
        self.thresholds = thresholds
        self.templates = templates or DEFAULT_TEMPLATES
        self.shap_summary = shap_summary

    def _shap_confirms(self, class_name: str) -> bool:
        if self.shap_summary is None:
            return True
        importance = self.shap_summary.get(class_name, {})
        if not importance:
            return True
        top_val = importance[max(importance, key=importance.get)]
        return top_val > 0.005

    def generate_report(self, fusion_probs: np.ndarray, lap_number: int = 1,
                         turn_number: Optional[int] = None) -> List[str]:
        """fusion_probs: (10,) sigmoid probabilities, FUSED_CLASS_NAMES order."""
        feedback = []
        turn_str = str(turn_number) if turn_number else "\u2014"

        for i, cls_name in enumerate(FUSED_CLASS_NAMES):
            prob = float(fusion_probs[i])
            thr = self.thresholds[i]
            if prob < thr:
                continue
            if not self._shap_confirms(cls_name):
                continue
            templates = self.templates.get(cls_name, [])
            if not templates:
                continue
            template = templates[i % len(templates)]
            feedback.append(template.format(lap=lap_number, turn=turn_str, prob=round(prob, 2)))

        return feedback

    def session_summary(self, all_probs: np.ndarray, lap_numbers: Optional[list] = None) -> dict:
        """all_probs: (N_windows, 10) fusion probabilities across all windows."""
        if len(all_probs) == 0:
            return {"event_counts": {n: 0 for n in FUSED_CLASS_NAMES},
                    "driving_persona": None, "feedback_items": [], "fer": 0.0}

        n_laps = max(lap_numbers) if lap_numbers else 1
        event_counts = {name: 0 for name in FUSED_CLASS_NAMES}
        feedback = []

        for window_idx, probs in enumerate(all_probs):
            lap = lap_numbers[window_idx] if lap_numbers else 1
            feedback.extend(self.generate_report(probs, lap_number=lap))
            for i, name in enumerate(FUSED_CLASS_NAMES):
                if probs[i] >= self.thresholds[i]:
                    event_counts[name] += 1

        aggressive_score = event_counts["Brake Locked"] + event_counts["Wheel Spin"] + event_counts["Rough Steering"]
        smooth_score = event_counts["Gentle Accel"]
        cautious_score = event_counts["Late Upshift"]

        if aggressive_score > smooth_score and aggressive_score > cautious_score:
            persona = "Aggressive"
        elif smooth_score >= aggressive_score and smooth_score >= cautious_score:
            persona = "Smooth"
        else:
            persona = "Cautious"

        fer = len(feedback) / max(n_laps, 1)

        return {
            "event_counts": event_counts,
            "driving_persona": persona,
            "feedback_items": feedback,
            "fer": round(fer, 2),
        }

    @staticmethod
    def thresholds_from_fusion_engine(fusion_engine) -> List[float]:
        """Pulls the per-class threshold list, in FUSED_CLASS_NAMES order,
        out of a loaded BimodalFusionEngine (which stores them keyed by
        label column name)."""
        return [fusion_engine.thresholds.get(col, 0.5) for col in FUSED_LABEL_COLS]
