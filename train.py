from src.machine_translation.mt_model import MT_Model
import argparse
from lt_sft.utils import get_mask, get_mask_granular, get_mask_layerwise

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Name of model checkpoint to load from HuggingFace")
    parser.add_argument("--out_model_name", type=str, required=True, help="Output Model Name")
    parser.add_argument("--metric", type=str, required=False, default="chrf", help="Metric to choose the best model on (Default: 'chrf') (Other: 'bleu')")
    parser.add_argument("--epochs", type=int, required=False, default=10, help="Number of epochs to train the model for (Default: 10)")
    parser.add_argument("--is_prompt", default=False, action=argparse.BooleanOptionalAction, help="Some models require a checkpoint (Default: False)")
    parser.add_argument("--extra_data_codes", nargs="*", type=str, default=[], help="Extra data to load for quechua (Default: []) (Options: 'quy', 'quz', 'que', 'bcktr', 'copied')")
    parser.add_argument("--is_multilingual", default=False, action=argparse.BooleanOptionalAction, help='Whether to train a multilingual model or just on Quechua (Default: False)')
    parser.add_argument("--with_dups", default=False, action=argparse.BooleanOptionalAction, help="Whether to include duplicate data but different dialect for Quechua (Default: False)")
    parser.add_argument("--push_to_hub", default=False, action=argparse.BooleanOptionalAction, help="Whether to push the model to the hub or not (Default: False)")
    parser.add_argument("--lt-sft", default=False, action=argparse.BooleanOptionalAction, help="Whether to use LT-SFT (Default: False)")
    parser.add_argument("--K", type=float, required=False, default=1, help="When using LT-SFT, K=proportion of parameters to leave unmasked (Default: 1)")
    parser.add_argument("--part", type=str, required=False, choices=["all", "encoder", "decoder"], default=None, help="(Optional) Which part of the model to apply sparse masking: 'all', 'encoder', or 'decoder'")
    parser.add_argument(
        "--encoder",
        nargs="+",
        type=float,
        metavar="P",
        help="Space-separated list of floats in [0,1]: per-encoder-layer keep-rates"
    )
    parser.add_argument(
        "--decoder",
        nargs="+",
        type=float,
        metavar="Q",
        help="Space-separated list of floats in [0,1]: per-decoder-layer keep-rates"
    )

    return parser

if __name__ == "__main__":

    parser = get_argument_parser()
    args = parser.parse_args()

    lt_sft_mask = None
    if args.encoder is not None or args.decoder is not None:
        if args.encoder:
            if any(p < 0.0 or p > 1.0 for p in args.encoder):
                parser.error("--encoder values must all be between 0.0 and 1.0")
        if args.decoder:
            if any(q < 0.0 or q > 1.0 for q in args.decoder):
                parser.error("--decoder values must all be between 0.0 and 1.0")

        print("Using new layerwise `get_mask_layerwise` method")
        layer_pct = {}
        if args.encoder is not None:
            layer_pct["encoder"] = args.encoder
        if args.decoder is not None:
            layer_pct["decoder"] = args.decoder

        lt_sft_mask = get_mask_layerwise(
            pretrained_model=args.checkpoint,
            layer_pct=layer_pct,
        )
    elif args.lt_sft:
        if args.part:
            print("Using `get_mask_granular` method")
            lt_sft_mask = get_mask_granular(
                pretrained_model=args.checkpoint,
                K_pct=args.K,
                part=args.part,
            )
        else:
            print("Using old `get_mask` method")
            lt_sft_mask = get_mask(K_pct=args.K)
        

    mt_model = MT_Model(
        checkpoint=args.checkpoint,
        out_model_name=args.out_model_name,
        que_data_codes=args.extra_data_codes,
        is_multilingual=args.is_multilingual,
        epochs=args.epochs,
        is_prompt=args.is_prompt,
        metric=args.metric,
        push_to_hub=args.push_to_hub,
        with_dups=args.with_dups,
        lt_sft_mask=lt_sft_mask,

    )
    mt_model.train()