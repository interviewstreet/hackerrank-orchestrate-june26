import logging
import re
from typing import List, Optional, Tuple

from .models import ClaimIntent, ClaimTarget

logger = logging.getLogger(__name__)


class ClaimParser:
    """Parses a claim conversation into a structured ClaimIntent."""

    ISSUE_TYPES = [
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
    ]

    OBJECT_PARTS = {
        "car": [
            "front_bumper",
            "rear_bumper",
            "door",
            "hood",
            "windshield",
            "side_mirror",
            "headlight",
            "taillight",
            "fender",
            "quarter_panel",
            "body",
        ],
        "laptop": [
            "screen",
            "keyboard",
            "trackpad",
            "hinge",
            "lid",
            "corner",
            "port",
            "base",
            "body",
        ],
        "package": [
            "box",
            "package_corner",
            "package_side",
            "seal",
            "label",
            "contents",
            "item",
        ],
    }

    ISSUE_KEYWORDS = {
        "glass_shatter": ["broken glass", "glass broke", "glass shattered", "shattered", "shatter", "shattering"],
        "broken_part": ["broken part", "broken piece", "broken", "rip", "ripped", "ripping"],
        "missing_part": ["missing part", "missing item", "missing", "lost", "gone"],
        "torn_packaging": ["torn packaging", "torn packet", "torn", "tear"],
        "crushed_packaging": ["crushed packaging", "crushed", "crush", "squashed", "smashed"],
        "water_damage": ["water damage", "waterlogged", "wet", "damp", "moisture", "leak"],
        "crack": ["crack", "cracked", "cracking", "fracture"],
        "dent": ["dent", "dented", "dents", "denting"],
        "scratch": ["scratch", "scratched", "scratches", "scraping", "scraped"],
        "stain": ["stain", "stained", "spotting"],
    }

    ISSUE_PRIORITY = [
        "glass_shatter",
        "broken_part",
        "missing_part",
        "crushed_packaging",
        "torn_packaging",
        "water_damage",
        "crack",
        "dent",
        "scratch",
        "stain",
    ]

    PART_KEYWORDS = {
        "front_bumper": ["front bumper", "front bumper area"],
        "rear_bumper": ["rear bumper", "back bumper", "rear bumper area"],
        "door": ["door", "door panel", "car door"],
        "hood": ["hood", "bonnet"],
        "windshield": ["windshield", "front glass", "wind screen", "windscreen"],
        "side_mirror": ["side mirror", "wing mirror", "side-mirror"],
        "headlight": ["headlight", "head light", "lamp"],
        "taillight": ["taillight", "tail light", "rear light"],
        "fender": ["fender", "wing"],
        "quarter_panel": ["quarter panel", "quarter-panel"],
        "body": ["body", "car body", "vehicle body"],
        "screen": ["screen", "display", "monitor"],
        "keyboard": ["keyboard", "keys"],
        "trackpad": ["trackpad", "track pad"],
        "hinge": ["hinge", "hinges"],
        "lid": ["lid", "top cover"],
        "corner": ["corner", "edge"],
        "port": ["port", "usb port", "charging port"],
        "base": ["base", "bottom", "chassis"],
        "box": ["box", "package box", "shipping box"],
        "package_corner": ["package corner", "box corner", "corner"],
        "package_side": ["package side", "box side", "side"],
        "seal": ["seal", "sealed", "tape"],
        "label": ["label", "shipping label", "package label"],
        "contents": ["contents", "inside", "inside the package", "items inside"],
        "item": ["item", "product", "parcel"],
    }

    SEVERITY_KEYWORDS = {
        "high": ["high", "severe", "serious", "major", "deep", "bad", "strong"],
        "medium": ["medium", "moderate", "noticeable", "visible"],
        "low": ["low", "minor", "small dent", "small scratch", "small crack", "small damage", "light", "slight", "mild"],
        "none": ["no damage", "no issue", "none"],
        "unknown": ["unknown", "not sure", "unclear", "cannot tell", "cant tell", "can't tell"],
    }

    INSTRUCTION_PATTERNS = [
        r"upload",
        r"attach",
        r"sent",
        r"please",
        r"add the photo",
        r"i attached",
    ]

    QUESTION_SPEAKERS = {"support", "agent", "rep", "assistant", "support:"}
    CUSTOMER_SPEAKERS = {"customer", "i", "me", "my"}

    AMBIGUITY_PATTERNS = [
        r"not sure",
        r"maybe",
        r"could be",
        r"i think",
        r"shayd",
        r"sakta hai",
        r"perhaps",
    ]

    def parse(self, user_claim: str, claim_object: str) -> ClaimIntent:
        """Parse the conversation text and return a ClaimIntent."""
        normalized_claim = self._normalize_text(user_claim)
        logger.debug("Normalized claim: %s", normalized_claim)

        segments = self._split_segments(user_claim)
        customer_segments = self._collect_customer_segments(segments)
        logger.debug("Customer segments: %s", customer_segments)

        if customer_segments:
            parse_text = " ".join(customer_segments)
        else:
            parse_text = self._choose_final_customer_segment(segments)

        issue = self._extract_issue_type(parse_text)
        part = self._extract_object_part(parse_text, claim_object)
        severity = self._extract_severity(parse_text)
        ambiguity_flags = self._detect_ambiguity(parse_text)
        untrusted_instruction_detected = self._detect_untrusted_instruction(normalized_claim)

        target = ClaimTarget(
            part=part,
            issue=issue,
            claimed_severity=severity,
            ambiguity=ambiguity_flags[0] if ambiguity_flags else None,
        )

        intent = ClaimIntent(
            declared_object=claim_object,
            targets=[target],
            ambiguity_flags=ambiguity_flags,
            untrusted_instruction_detected=untrusted_instruction_detected,
        )

        logger.debug("ClaimIntent created: %s", intent)
        return intent

    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching by lowering and normalizing spacing."""
        normalized = text.lower().strip()
        normalized = normalized.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _split_segments(self, text: str) -> List[Tuple[str, str]]:
        """Split conversation text into speaker segments."""
        segments: List[Tuple[str, str]] = []
        for raw_segment in text.split("|"):
            segment = raw_segment.strip()
            if not segment:
                continue
            if ":" in segment:
                speaker, _, utterance = segment.partition(":")
                segments.append((speaker.strip().lower(), utterance.strip()))
            else:
                segments.append(("unknown", segment.strip()))
        return segments

    def _is_customer_speaker(self, speaker: str) -> bool:
        """Return True if the speaker is likely the customer."""
        normalized = speaker.lower().strip()
        return any(keyword == normalized or keyword in normalized for keyword in self.CUSTOMER_SPEAKERS)

    def _is_support_speaker(self, speaker: str) -> bool:
        """Return True if the speaker is likely a support agent or system speaker."""
        normalized = speaker.lower().strip()
        return any(keyword == normalized or keyword in normalized for keyword in self.QUESTION_SPEAKERS)

    def _collect_customer_segments(self, segments: List[Tuple[str, str]]) -> List[str]:
        """Collect customer utterances in order while excluding support questions."""
        collected: List[str] = []
        for speaker, utterance in segments:
            if self._is_support_speaker(speaker):
                continue
            if self._is_customer_speaker(speaker) or speaker == "unknown":
                collected.append(self._normalize_text(utterance))
        return collected

    def _choose_final_customer_segment(self, segments: List[Tuple[str, str]]) -> str:
        """Choose a fallback segment when no explicit customer utterance exists."""
        for speaker, utterance in reversed(segments):
            if self._is_customer_speaker(speaker):
                return self._normalize_text(utterance)
        for speaker, utterance in reversed(segments):
            if not self._is_support_speaker(speaker):
                return self._normalize_text(utterance)
        if segments:
            return self._normalize_text(segments[-1][1])
        return ""

    def _find_last_pattern_match(self, normalized: str, patterns: List[str]) -> Optional[Tuple[str, int]]:
        """Return the last matching pattern and its position in normalized text."""
        last_match: Optional[Tuple[str, int]] = None
        for pattern in patterns:
            for match in re.finditer(rf"\b{re.escape(pattern)}\b", normalized):
                last_match = (pattern, match.start())
        return last_match

    def _extract_issue_type(self, text: str) -> Optional[str]:
        """Extract the latest matching issue type from the parsed customer text."""
        normalized = self._normalize_text(text)
        matched_issues = {}
        for issue_type, patterns in self.ISSUE_KEYWORDS.items():
            for pattern in patterns:
                for match in re.finditer(rf"\b{re.escape(pattern)}\b", normalized):
                    position = match.start()
                    if issue_type not in matched_issues or position >= matched_issues[issue_type]:
                        matched_issues[issue_type] = position
        if not matched_issues:
            logger.debug("No issue_type match found in text=%s", normalized)
            return None
        # Prefer the highest-priority matched issue type across the entire customer text.
        # If multiple issues share the same priority, use the latest mention of that priority.
        for issue_type in self.ISSUE_PRIORITY:
            if issue_type in matched_issues:
                logger.debug("Selected issue_type=%s from text=%s", issue_type, normalized)
                return issue_type
        selected = min(matched_issues, key=lambda key: matched_issues[key])
        logger.debug("Selected fallback issue_type=%s from text=%s", selected, normalized)
        return selected

    def _extract_object_part(self, text: str, claim_object: str) -> Optional[str]:
        """Extract the last matching object part from the customer text."""
        normalized = self._normalize_text(text)
        part_candidates = self.OBJECT_PARTS.get(claim_object, [])
        selected_part: Optional[str] = None
        last_position = -1
        for part in part_candidates:
            patterns = self.PART_KEYWORDS.get(part, [])
            for pattern in patterns:
                for match in re.finditer(rf"\b{re.escape(pattern)}\b", normalized):
                    if match.start() >= last_position:
                        selected_part = part
                        last_position = match.start()
        logger.debug("Extracted object_part=%s from text=%s", selected_part, normalized)
        return selected_part

    def _extract_severity(self, text: str) -> Optional[str]:
        """Extract the last matching severity from the customer text."""
        normalized = self._normalize_text(text)
        selected_severity: Optional[str] = None
        last_position = -1
        for severity, patterns in self.SEVERITY_KEYWORDS.items():
            for pattern in patterns:
                for match in re.finditer(rf"\b{re.escape(pattern)}\b", normalized):
                    if match.start() >= last_position:
                        selected_severity = severity
                        last_position = match.start()
        logger.debug("Extracted severity=%s from text=%s", selected_severity, normalized)
        return selected_severity

    def _detect_ambiguity(self, text: str) -> List[str]:
        """Detect ambiguity signals in the parsed customer text."""
        normalized = self._normalize_text(text)
        flags: List[str] = []
        for pattern in self.AMBIGUITY_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b", normalized):
                flags.append("ambiguous_claim")
                break
        logger.debug("Ambiguity flags=%s from text=%s", flags, normalized)
        return flags

    def _detect_untrusted_instruction(self, text: str) -> bool:
        """Detect if the claim contains instruction-like or untrusted text."""
        normalized = self._normalize_text(text)
        for pattern in self.INSTRUCTION_PATTERNS:
            if re.search(rf"\b{re.escape(pattern)}\b", normalized):
                logger.debug("Detected instruction-like text: %s", pattern)
                return True
        return False
