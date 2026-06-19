import pytest

from src.claim_parser import ClaimParser
from src.models import ClaimIntent, ClaimTarget


@pytest.fixture
def parser() -> ClaimParser:
    return ClaimParser()


def test_claim_parser_extracts_final_customer_segment_and_issue():
    user_claim = (
        "Customer: I am opening a claim for my windshield. | Support: What happened? | "
        "Customer: A small stone hit it while I was driving and now there is a crack spreading from that spot. | "
        "Support: Is the car otherwise okay? | Customer: Yes, this is only about the front glass."
    )

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert isinstance(intent, ClaimIntent)
    assert intent.declared_object == "car"
    assert len(intent.targets) == 1
    assert intent.targets[0].issue == "crack"
    assert intent.targets[0].part == "windshield"
    assert intent.targets[0].claimed_severity is None
    assert intent.targets[0].ambiguity is None
    assert intent.untrusted_instruction_detected is False


def test_claim_parser_detects_severity_and_instruction():
    user_claim = (
        "Customer: Need to file a car damage claim. | Agent: What part of the car? | "
        "Customer: Door. | Agent: Scratch, dent, or paint issue? | Customer: A deep dent on the door panel. It was not there before."
    )

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].issue == "dent"
    assert intent.targets[0].part == "door"
    assert intent.targets[0].claimed_severity == "high"
    assert intent.untrusted_instruction_detected is False


def test_claim_parser_handles_multilingual_text():
    user_claim = (
        "Customer: Parking lot mein meri car ko scrape lag gaya. | Support: Aap kis type ka damage report karna chahte hain? | "
        "Customer: Front side par mark aa gaya hai, bumper ke upar. | Support: Light damage hai ya body par scratch? | "
        "Customer: Light theek hai, front bumper par scratch hai. Photos upload kar diye hain."
    )

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].issue == "scratch"
    assert intent.targets[0].part == "front_bumper"
    assert intent.targets[0].claimed_severity == "low"
    assert intent.untrusted_instruction_detected is True


def test_claim_parser_returns_none_when_no_match():
    user_claim = "Customer: I found new damage on my car after it was parked outside overnight."

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].issue is None
    assert intent.targets[0].part is None
    assert intent.targets[0].claimed_severity is None


def test_claim_parser_extracts_headlight_and_hyd_resolution():
    user_claim = (
        "Customer: My headlight is broken. | Agent: Can you share the image? | "
        "Customer: The left headlight has a crack and the glass is shattered."
    )
    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].part == "headlight"
    assert intent.targets[0].issue == "glass_shatter"


def test_claim_parser_extracts_windshield():
    user_claim = "Customer: The windshield has a big crack from the stone."

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].part == "windshield"
    assert intent.targets[0].issue == "crack"


def test_claim_parser_extracts_side_mirror():
    user_claim = "Customer: My side mirror is broken and hanging off."

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].part == "side_mirror"
    assert intent.targets[0].issue == "broken_part"


def test_claim_parser_extracts_hood():
    user_claim = "Customer: There is a dent on the hood after the accident."

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].part == "hood"
    assert intent.targets[0].issue == "dent"


def test_claim_parser_extracts_laptop_screen_and_keyboard():
    user_claim = (
        "Customer: The laptop screen is cracked. | Support: Does the keyboard also work? | "
        "Customer: Keyboard keys are fine but the screen is damaged."
    )

    intent = ClaimParser().parse(user_claim, claim_object="laptop")

    assert intent.targets[0].part == "screen"
    assert intent.targets[0].issue == "crack"


def test_claim_parser_extracts_hinge():
    user_claim = "Customer: The laptop hinge is broken and won\'t close properly."

    intent = ClaimParser().parse(user_claim, claim_object="laptop")

    assert intent.targets[0].part == "hinge"
    assert intent.targets[0].issue == "broken_part"


def test_claim_parser_extracts_package_box_and_corner():
    user_claim = "Customer: The box arrived crushed and the package corner is torn."

    intent = ClaimParser().parse(user_claim, claim_object="package")

    assert intent.targets[0].part == "package_corner"
    assert intent.targets[0].issue == "crushed_packaging"


def test_claim_parser_extracts_package_seal():
    user_claim = "Customer: The package seal is ripped open when it arrived."

    intent = ClaimParser().parse(user_claim, claim_object="package")

    assert intent.targets[0].part == "seal"
    assert intent.targets[0].issue == "broken_part"


def test_claim_parser_extracts_water_damage():
    user_claim = "Customer: My package shows water damage and it is damp inside."

    intent = ClaimParser().parse(user_claim, claim_object="package")

    assert intent.targets[0].issue == "water_damage"


def test_claim_parser_extracts_missing_part():
    user_claim = "Customer: The item is missing from the package."

    intent = ClaimParser().parse(user_claim, claim_object="package")

    assert intent.targets[0].issue == "missing_part"


def test_claim_parser_detects_broken_part():
    user_claim = "Customer: The car has a broken part on the bumper."

    intent = ClaimParser().parse(user_claim, claim_object="car")

    assert intent.targets[0].issue == "broken_part"


def test_claim_parser_detects_ambiguity_phrases():
    for text in [
        "Customer: I think the damage is on the door.",
        "Customer: Maybe it is the hood.",
        "Customer: I am not sure about the headlight.",
    ]:
        intent = ClaimParser().parse(text, claim_object="car")
        assert intent.targets[0].ambiguity == "ambiguous_claim"
        assert "ambiguous_claim" in intent.ambiguity_flags
