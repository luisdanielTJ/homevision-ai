PROMPT_TEMPLATE = """You are an assistant for Home Depot Canada store associates.
A customer has been detected in the store who may need help.

Details:
- Camera zone: {camera_id}
- Time standing in aisle: {dwell_time:.0f} seconds
- Head turns detected: {head_turns}
- Object raised for inspection: {object_raised}

Write a short, friendly 1-sentence notification for the nearest store associate.
Be specific and actionable. Do not mention AI or cameras."""


class GeminiEnricher:
    def __init__(
        self,
        project_id: str = "",
        region: str = "",
        model: str = "gemini-1.5-flash",
        mock: bool = False,
    ) -> None:
        self._mock = mock
        if not mock:
            import vertexai
            from vertexai.generative_models import GenerativeModel

            vertexai.init(project=project_id, location=region)
            self._model = GenerativeModel(model)

    def enrich(self, camera_id: str, dwell_time: float, head_turns: int, object_raised: bool) -> str:
        if self._mock:
            return (
                f"Customer in zone {camera_id} has been standing for {dwell_time:.0f}s "
                f"and appears to need assistance with a product."
            )
        prompt = PROMPT_TEMPLATE.format(
            camera_id=camera_id,
            dwell_time=dwell_time,
            head_turns=head_turns,
            object_raised=object_raised,
        )
        response = self._model.generate_content(prompt)
        return response.text.strip()
