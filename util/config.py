from pydantic import BaseModel, computed_field


class Config(BaseModel):
    GRAVITY: float = 9.81
    L: float = 1.0
    M: float = 1.0
    TIMESPAN: float = 8.0
    NUM_SAMPLES: int = 600
    SAMPLING_POSITIONS: list[float] = [1.0, 0.8, 0.6, 0.4]

    @computed_field
    def DT(self) -> float:
        return self.TIMESPAN / self.NUM_SAMPLES

    @computed_field
    def NUM_POINTS(self) -> int:
        return len(self.SAMPLING_POSITIONS)

    def __hash__(self):
        return hash(
            (
                self.GRAVITY,
                self.L,
                self.M,
                self.TIMESPAN,
                self.NUM_SAMPLES,
                tuple(self.SAMPLING_POSITIONS),
            )
        )
