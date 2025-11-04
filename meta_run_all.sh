

CONFIG_FILE="meta_config.txt"
TAG_SUFFIX="03-11-25"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-c|--config <config_file>]"
            exit 0
            ;;
        -t|--tag)
            TAG_SUFFIX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

while read -r train_path val_path vis_path cont_path forecast alpha tag epochs; do
    [[ -z "$train_path" || "$train_path" =~ ^# ]] && continue

    echo "Running meta experiment with training: $train_path, validation: $val_path, visualisation: $vis_path, continuity: $cont_path, forecast: $forecast, alpha: $alpha, tag: $tag, epochs: $epochs"
    ./run_all_experiments.sh \
        --training-file "$train_path" \
        --validation-file "$val_path" \
        --visualisation-file "$vis_path" \
        --continuity-file "$cont_path" \
        --forecast "$forecast" \
        --alpha "$alpha" \
        --tag "${tag}_$TAG_SUFFIX" \
        --epochs "$epochs"
done < "$CONFIG_FILE"