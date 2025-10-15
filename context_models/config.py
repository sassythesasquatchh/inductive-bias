from context_models import encoders, dynamics, decoders

encoder_dict = {
    "informed": encoders.InformedEncoder,
    "unstructured": encoders.UnstructuredEncoder,
    "cnn": encoders.CNNEncoder,
}

dynamics_dict = {
    "informed": dynamics.InformedDynamics,
    "hybrid": dynamics.HybridDynamics,
    "unstructured": dynamics.UnstructuredDynamics,
}

decoder_dict = {
    "informed": decoders.InformedDecoder,
    "unstructured": decoders.UnstructuredDecoder,
    "cnn": decoders.CNNDecoder,
}
