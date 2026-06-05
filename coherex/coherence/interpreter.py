from coherex.config import CONFIG


class MotionInterpreter:
    """
    Interprets continuous MCV into qualitative motion states.
    """

    def __init__(self, low=None, high=None, config=None):
        cfg = config or CONFIG
        self.low = low if low is not None else cfg.motion.interpreter_low
        self.high = high if high is not None else cfg.motion.interpreter_high

    def interpret(self, mcv):
        if mcv < self.low:
            return "COHERENT"
        elif mcv < self.high:
            return "SUSPICIOUS"
        else:
            return "INCONSISTENT"
