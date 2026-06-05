# coherex/integrity/base_agent.py


class IntegrityAgent:
    """
    Base class for all integrity agents.

    Each agent must return a coherence score in range [0, 1], where:
        1.0 = fully coherent (no anomaly detected)
        0.0 = highly inconsistent (strong anomaly signal)

    All subclasses must implement evaluate().
    """

    def evaluate(self, context):
        """
        Evaluate coherence from the given context.

        Args:
            context: Agent-specific input data.

        Returns:
            float: Coherence score in [0.0, 1.0].
        """
        raise NotImplementedError("Agent must implement evaluate()")
