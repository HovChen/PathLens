import torch
import transformers
from models.llava.model import LlavaPhiForCausalLM
from models.llava.peft import LoraConfig, get_peft_model
from utils.utils import find_all_linear_names, add_special_tokens_and_resize_model, load_weights, com_vision_args
from utils.logger import logger

def create_model_and_tokenizer(args):
    logger.info("Creating model and tokenizer...")

    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)

    model = LlavaPhiForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name_or_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=model_dtype
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    num_new_tokens = add_special_tokens_and_resize_model(tokenizer, model)
    logger.debug(f"Number of new tokens added for reasoning task: {num_new_tokens}")
    with torch.no_grad():
        emb = model.get_input_embeddings()
        lt_id = tokenizer.convert_tokens_to_ids("<")
        gt_id = tokenizer.convert_tokens_to_ids(">")
        avg_vec = 0.5 * (emb.weight[lt_id] + emb.weight[gt_id])

        for tok in ["<think>", "</think>", "<answer>", "</answer>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            emb.weight[tid].copy_(avg_vec)


    lora_config = LoraConfig(
        r=args.hlora_r,
        lora_alpha=args.hlora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=args.hlora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
        lora_nums=args.hlora_nums,
        modules_to_save=["embed_tokens", "lm_head"] if not args.infer else None,
    )
    model = get_peft_model(model, lora_config)

    if not args.infer:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params}, All params: {all_params}, Ratio: {trainable_params/all_params:.2%}")

    com_vision_args.model_name_or_path = args.model_name_or_path
    com_vision_args.vision_tower = args.vit_path
    com_vision_args.version = args.instruct_template

    model.get_model().initialize_vision_modules(model_args=com_vision_args)
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=model_dtype)

    if not args.infer:
        model.gradient_checkpointing_enable()
        model.config.use_cache = True
        model.enable_input_require_grads()

    model = load_weights(model, args.hlora_path, args.vocab_proj_path)

    model = model.to(model_dtype)
    return model, tokenizer