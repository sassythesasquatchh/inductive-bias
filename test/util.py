from context_models.decoders import InformedDecoder
from context_models.dynamics import InformedDynamics
from context_models.encoders import InformedEncoder
from context_models.model import ContextModel
from util.config import Config

DEFAULT_CONFIG = Config()

DEFAULT_TRAIN_PATH = "data/normal_training_5.pkl"
DEFAULT_VAL_PATH = "data/validation_5.pkl"
DEFAULT_VISUALISATION_PATH = "data/visualisation.pkl"
DEFAULT_CONTINUITY_PATH = "data/continuity_test.pkl"

DEFAULT_MODEL = ContextModel(
    encoder=InformedEncoder(config=DEFAULT_CONFIG, context=1),
    dynamics=InformedDynamics(config=DEFAULT_CONFIG),
    decoder=InformedDecoder(config=DEFAULT_CONFIG),
    forecast=2,
)
