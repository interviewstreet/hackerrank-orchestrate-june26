import os
import base64
from config import ClaimEvaluationSchema

def encode_image_to_base64(image_path: str) -> str:
    """Helper to convert a local image into a base64 string for API transmission."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class MultiModalVLMClient:
    def __init__(self, provider: str = "gemini"):
        """
        provider: 'gemini' or 'claude'
        """
        self.provider = provider.lower()
        
        if self.provider == "gemini":
            # Initialize Gemini GenAI Client
            from google import genai
            from google.genai import types
            self.client = genai.Client() # Automatically picks up GEMINI_API_KEY
            self.model_name = "gemini-2.5-flash"  # Or gemini-2.5-flash for faster runs
            self.types = types
            
        elif self.provider == "claude":
            # Initialize Anthropic Client
            import anthropic
            self.client = anthropic.Anthropic() # Automatically picks up ANTHROPIC_API_KEY
            self.model_name = "claude-3-5-sonnet-20241022"

    def evaluate_claim(self, system_prompt: str, user_prompt: str, images: list) -> ClaimEvaluationSchema:
        """
        Sends the text and images to the selected VLM and enforces the Pydantic schema output.
        images: list of dicts with [{'path': '...', 'id': '...'}]
        """
        if self.provider == "gemini":
            return self._call_gemini(system_prompt, user_prompt, images)
        elif self.provider == "claude":
            return self._call_claude(system_prompt, user_prompt, images)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_gemini(self, system_prompt: str, user_prompt: str, images: list) -> ClaimEvaluationSchema:
        # Prepare contents array with text prompt
        contents = [user_prompt]
        
        # Append images as binary data blocks
        for img in images:
            with open(img['path'], 'rb') as f:
                img_bytes = f.read()
            contents.append(
                self.types.Part.from_bytes(
                    data=img_bytes,
                    mime_type="image/jpeg"
                )
            )
            
        # Call Gemini with Structured Outputs enforcement
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self.types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=ClaimEvaluationSchema,
                temperature=0.1
            ),
        )
        
        # The SDK automatically handles validation and gives us a parsed object via parsed or text
        return ClaimEvaluationSchema.model_validate_json(response.text)

    def _call_claude(self, system_prompt: str, user_prompt: str, images: list) -> ClaimEvaluationSchema:
        import json
        
        # Format messages structure for Claude
        message_content = []
        
        # Add visual image blocks
        for img in images:
            base64_data = encode_image_to_base64(img['path'])
            message_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64_data
                }
            })
            
        # Append the analytical textual prompts
        # Claude expects structural fields via tools or prompt-based JSON enforcement
        schema_json = json.dumps(ClaimEvaluationSchema.model_json_schema(), indent=2)
        full_user_prompt = f"{user_prompt}\n\nYou MUST return your absolute appraisal as a raw JSON object matching this exact structural schema:\n{schema_json}"
        
        message_content.append({
            "type": "text",
            "text": full_user_prompt
        })
        
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": message_content}]
        )
        
        # Parse text content safely back to Pydantic
        raw_text = response.content[0].text
        # Strip potential markdown code fences if Claude wraps it
        if raw_text.startswith("```json"):
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1].split("```")[0].strip()
            
        return ClaimEvaluationSchema.model_validate_json(raw_text)