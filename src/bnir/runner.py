import copy

from tevatron.arguments import DataArguments
from tevatron.data import QPCollator, TrainDataset
from tevatron.datasets import HFTrainDataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from bnir.arguments import ModelArguments, TrainingArguments
from bnir.encoders import BayesianDenseEncoder, DenseEncoder
from bnir.trainers import BayesianTevatronTrainer, TevatronTrainer


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    set_seed(train_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    train_dataset = HFTrainDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        cache_dir=data_args.data_cache_dir or model_args.cache_dir,
    )
    train_dataset = TrainDataset(data_args, train_dataset.process(), tokenizer)

    if model_args.model_type == "vanilla":
        model = DenseEncoder.build(
            model_args,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        trainer = TevatronTrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=QPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len,
            ),
        )
    elif model_args.model_type == "vi":
        model = BayesianDenseEncoder.build(
            model_args,
            train_args,
            len(train_dataset.train_data),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        base_qry_model = AutoModel.from_config(
            config=config,
        )  # Randomly initialized backbone, its values will be filled in on the fly.
        if model_args.weight_sharing:
            base_psg_model = base_qry_model
        else:
            base_psg_model = copy.deepcopy(base_qry_model)
        prior_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
        trainer = BayesianTevatronTrainer(
            base_qry_model,
            base_psg_model,
            prior_model,
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            data_collator=QPCollator(
                tokenizer,
                max_p_len=data_args.p_max_len,
                max_q_len=data_args.q_max_len,
            ),
        )
    else:
        raise NotImplementedError(f"Algorithm {model_args.model_type} not supported.")
    train_dataset.trainer = trainer

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(train_args.output_dir)


if __name__ == "__main__":
    main()
