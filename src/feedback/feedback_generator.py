"""
feedback_generator.py — Turns fusion head predictions + SHAP attribution
into the post-session coaching report, per proposal §3.5.5.

Logic gate: an event fires a feedback item ONLY IF
    prob > threshold  AND  SHAP confirms the primary contributing feature
The SHAP confirmation prevents feedback being generated when the model is
uncertain — if SHAP shows the prediction was driven by a low-confidence
noisy signal, the template is suppressed.

Ported from `final-cnn-module-codes.ipynb` ("16.4 Late Fusion Pipeline",
SECTION 6: Feedback generator).
"""

import numpy as np

from src.fusion.fusion_constants import CLASS_NAMES
from src.fusion.late_fusion_head import LateFusionHead
from src.feedback.template_library import DEFAULT_TEMPLATES
from src.feedback.persona_classifier import classify_persona


class FeedbackGenerator:
    """Takes fusion head predictions + SHAP attribution and generates the post-session coaching report."""

    def __init__(self, fusion_model: LateFusionHead, thresholds: list,
                 templates: dict = None, shap_summary: dict = None):
        self.model = fusion_model
        self.thresholds = thresholds        # list of 10 floats
        self.templates = templates or DEFAULT_TEMPLATES
        self.shap_summary = shap_summary    # used for SHAP confirmation gate

    def _shap_confirms(self, class_name: str, x: np.ndarray) -> bool:
        """
        Checks that the top-contributing feature for this class has a
        magnitude consistent with a genuine signal. If no SHAP summary is
        loaded, always returns True (gate disabled).
        """
        if self.shap_summary is None:
            return True
        importance = self.shap_summary.get(class_name, {})
        if not importance:
            return True
        top_feat = max(importance, key=importance.get)
        top_val = importance[top_feat]
        return top_val > 0.005  # suppress if top feature's SHAP is below noise floor

    def generate_report(self, fusion_probs: np.ndarray, lap_number: int = 1, turn_number: int = None) -> list:
        """
        Generate a list of feedback strings for one session window.

        Returns:
            List of plain-language feedback strings. Empty list = no issues.
        """
        feedback = []
        turn_str = str(turn_number) if turn_number else "-"

        for i, cls_name in enumerate(CLASS_NAMES):
            prob = float(fusion_probs[i])
            thr = self.thresholds[i]

            if prob < thr:
                continue
            if not self._shap_confirms(cls_name, fusion_probs):
                continue

            templates = self.templates.get(cls_name, [])
            if not templates:
                continue

            template = templates[i % len(templates)]
            feedback.append(template.format(lap=lap_number, turn=turn_str, prob=round(prob, 2)))

        return feedback

    def session_summary(self, all_probs: np.ndarray, lap_numbers: list = None) -> dict:
        """
        Aggregate predictions across a full session into a coaching report.

        Returns:
            {event_counts, driving_persona, feedback_items, fer}
        """
        n_laps = max(lap_numbers) if lap_numbers else 1
        event_counts = {name: 0 for name in CLASS_NAMES}
        feedback = []

        for window_idx, probs in enumerate(all_probs):
            lap = lap_numbers[window_idx] if lap_numbers else 1
            items = self.generate_report(probs, lap_number=lap)
            feedback.extend(items)
            for i, name in enumerate(CLASS_NAMES):
                if probs[i] >= self.thresholds[i]:
                    event_counts[name] += 1

        persona = classify_persona(event_counts)
        fer = len(feedback) / max(n_laps, 1)

        print(f"\n{'='*55}\n  POST-SESSION COACHING REPORT\n{'='*55}")
        print(f"  Driving Persona : {persona}")
        print(f"  Feedback items  : {len(feedback)}")
        print(f"  FER (items/lap) : {fer:.2f}")
        print("\n  Event counts:")
        for name, count in event_counts.items():
            bar = "#" * min(count // 10, 20)
            print(f"    {name:<24} {count:>5}  {bar}")
        print("\n  Feedback:")
        for item in feedback[:15]:
            print(f"    - {item}")
        if len(feedback) > 15:
            print(f"    ... and {len(feedback)-15} more items")
        print(f"{'='*55}\n")

        return {"event_counts": event_counts, "driving_persona": persona,
                "feedback_items": feedback, "fer": round(fer, 2)}
