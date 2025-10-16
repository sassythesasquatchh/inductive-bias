from context_models import decoders, dynamics, encoders

encoder_dict = {
    "unstructured": encoders.UnstructuredEncoder,
    "informed": encoders.InformedEncoder,
    "cnn": encoders.CNNEncoder,
}

dynamics_dict = {
    "informed": dynamics.InformedDynamics,
    "unstructured": dynamics.UnstructuredDynamics,
    "hybrid": dynamics.HybridDynamics,
}

decoder_dict = {
    "unstructured": decoders.UnstructuredDecoder,
    "informed": decoders.InformedDecoder,
    "cnn": decoders.CNNDecoder,
}
