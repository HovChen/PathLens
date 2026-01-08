import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import tokenizers
from PIL import Image
import argparse
from utils.utils import expand2square
from utils.logger import logger
from packaging import version
from models.model_setup import create_model_and_tokenizer
from models.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from models.llava import conversation as conversation_lib
from models.llava.mm_utils import tokenizer_image_token
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='microsoft/Phi-3-mini-4k-instruct')
    parser.add_argument('--dtype', type=str, default='FP32')
    parser.add_argument('--attn_implementation', type=str, default='flash_attention_2')
    parser.add_argument('--hlora_r', type=int, default=16)
    parser.add_argument('--hlora_alpha', type=int, default=32)
    parser.add_argument('--hlora_dropout', type=float, default=0.0)
    parser.add_argument('--hlora_nums', type=int, default=4)
    parser.add_argument('--instruct_template', type=str, default='phi3_instruct')
    parser.add_argument('--vit_path', type=str, default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--hlora_path', type=str, default=None)
    parser.add_argument('--vocab_proj_path', type=str, default=None)
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--img_path', type=str, default=None)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--infer', action='store_true', default=True)
    return parser.parse_args()

def infer():
    args = get_args()
    model_dtype = torch.float32 if args.dtype == 'FP32' else (torch.float16 if args.dtype == 'FP16' else torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.set_default_device(device)
    except AttributeError:
        pass

    model, tokenizer = create_model_and_tokenizer(args)
    model.to(device)
    model.eval()
    question = args.question
    img_path = args.img_path

    if img_path:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        qs = question
    conv = conversation_lib.conv_templates[args.instruct_template].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(device).unsqueeze_(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    if img_path:
        image = Image.open(img_path).convert('RGB')
        image = expand2square(image, tuple(int(x*255) for x in model.get_vision_tower().image_processor.image_mean))
        image_tensor = model.get_vision_tower().image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze_(0).to(dtype=model_dtype, device=device, non_blocking=True)

    with torch.inference_mode():
        output_ids = model.base_model.model.generate(
        input_ids,
        images=image_tensor if img_path else None,
        image_sizes=image.size if img_path else None,
        attention_mask=attention_mask,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True)
    
    full_txt = tokenizer.decode(output_ids[0],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True)
    # m = re.search(r"<think>.*?</answer>", full_txt, flags=re.DOTALL)
    # response = m.group(0) if m else full_txt

    # token_list = tokenizer.convert_ids_to_tokens(output_ids[0].tolist())
    
    print(f'Question: {question}')
    print(f'PathLens: {full_txt}')
    # print(f'Token List:{token_list}')


if __name__ == "__main__":
    infer()