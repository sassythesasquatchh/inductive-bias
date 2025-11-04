from context_models_3 import decoders, dynamics, encoders

encoder_dict = {
    "unstructured": encoders.UnstructuredEncoder,
    "informed": encoders.InformedEncoder,
    "cnn": encoders.CNNEncoder,
    "identity": encoders.IdentityEncoder,
}

dynamics_dict = {
    "informed": dynamics.InformedDynamics,
    "unstructured": dynamics.UnstructuredDynamics,
    "hybrid": dynamics.HybridDynamics,
    "hybrid_alt": dynamics.HybridDynamics2,
}

decoder_dict = {
    "unstructured": decoders.UnstructuredDecoder,
    "informed": decoders.InformedDecoder,
    "identity": decoders.IdentityDecoder,
}
