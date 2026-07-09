class LLMClient:
    """Phase 2 will add real LLM providers behind this interface."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("LLM generation is implemented in Phase 2.")

