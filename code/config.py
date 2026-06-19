from pydantic import BaseModel, Field
from typing import Literal

# Define strict allowed value constraints from the problem statement
CLAIM_STATUS_TYPE = Literal["supported", "contradicted", "not_enough_information"]

ISSUE_TYPE = Literal[
    "dent", "scratch", "crack", "glass_shatter", "broken_part", 
    "missing_part", "torn_packaging", "crushed_packaging", 
    "water_damage", "stain", "none", "unknown"
]

OBJECT_PART_TYPE = Literal[
    # Car parts
    "front_bumper", "rear_bumper", "door", "hood", "windshield", 
    "side_mirror", "headlight", "taillight", "fender", "quarter_panel", "body",
    # Laptop parts
    "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port", "base",
    # Package parts
    "box", "package_corner", "package_side", "seal", "label", "contents", "item",
    # Catch-all
    "unknown"
]

SEVERITY_TYPE = Literal["none", "low", "medium", "high", "unknown"]

class ClaimEvaluationSchema(BaseModel):
    """
    Strict structural alignment with the requested final output schema.
    """
    evidence_standard_met: bool = Field(description="true if the image set is sufficient to evaluate the claim; otherwise false")
    evidence_standard_met_reason: str = Field(description="Short, concise reason for the evidence decision")
    risk_flags: str = Field(description="Semicolon-separated risk flags matching allowed values, or 'none'")
    issue_type: ISSUE_TYPE = Field(description="Visible issue type from the allowed values list")
    object_part: OBJECT_PART_TYPE = Field(description="Relevant object part matching the claim object type")
    claim_status: CLAIM_STATUS_TYPE = Field(description="Final decision: supported, contradicted, or not_enough_information")
    claim_status_justification: str = Field(description="Concise image-grounded explanation mentioning relevant image IDs")
    supporting_image_ids: str = Field(description="Image IDs supporting the decision separated by semicolons, or 'none'")
    valid_image: bool = Field(description="true if the image set is usable for automated review; otherwise false")
    severity: SEVERITY_TYPE = Field(description="Estimated damage severity: none, low, medium, high, or unknown")