#!/bin/bash
set -e

# Launch a Gradio chat UI for a fine-tuned LLaVA model.
#
# Usage:
#   ./scripts/run_llava_ui.sh /path/to/lora_checkpoint
#
#   PORT=7860 DEVICE=0 ./scripts/run_llava_ui.sh /path/to/lora_checkpoint
#
# The checkpoint should be the LoRA output dir from run_finetune_llava.sh.

MODEL_PATH=${1:?Usage: $0 /path/to/lora_checkpoint}

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

MODEL_BASE=${MODEL_BASE:-liuhaotian/llava-v1.5-7b}
DEVICE=${DEVICE:-0}
PORT=${PORT:-7860}
SHARE=${SHARE:-False}
DTYPE=${DTYPE:-float16}

export CUDA_VISIBLE_DEVICES=$DEVICE

cd /mnt/data/code/KUEA/LLaVA

echo "=== LLaVA Gradio UI ==="
echo "Model:  $MODEL_PATH"
echo "Base:   $MODEL_BASE"
echo "Device: $DEVICE"
echo "Port:   $PORT"
echo ""

python -c "
import sys
sys.path.insert(0, '.')

import torch
import gradio as gr
from PIL import Image

from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

model_path = '${MODEL_PATH}'
model_base = '${MODEL_BASE}'
dtype = '${DTYPE}'
port = ${PORT}
share = ${SHARE}

model_name = get_model_name_from_path(model_path)
model, image_processor, tokenizer, context_len = load_pretrained_model(
    model_path, model_base, model_name, dtype=dtype,
)
model.eval()

def respond(message, image, history, temperature, max_tokens):
    conv = conv_templates['llava_v1'].copy()

    if image is not None:
        image = image.convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        image_size = image.size
    else:
        image_tensor = None
        image_size = None

    # Replay history
    for msg_dict in history:
        content = msg_dict['content']
        if isinstance(content, list):
            content = ' '.join(c['text'] for c in content if isinstance(c, dict) and 'text' in c)
        if msg_dict['role'] == 'user':
            conv.append_message(conv.roles[0], content)
        else:
            conv.append_message(conv.roles[1], content)

    # Prepend image token whenever an image is provided
    if image is not None:
        message = DEFAULT_IMAGE_TOKEN + '\n' + message

    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_tokens,
            use_cache=True,
        )

    output = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

    return output

with gr.Blocks(title='LLaVA - KUEA') as demo:
    gr.Markdown('# LLaVA Chat (KUEA-aligned)')
    gr.Markdown(f'Model: \`{model_path}\`')

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(type='pil', label='Image')
            temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.1, label='Temperature')
            max_tokens = gr.Slider(64, 1024, value=512, step=64, label='Max tokens')
            clear_btn = gr.Button('Clear')
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(height=600, label='Chat')
            msg = gr.Textbox(placeholder='Ask about the image...', show_label=False)

    def user_submit(message, image, history):
        reply = respond(message, image, history, temperature.value if hasattr(temperature, 'value') else 0.2, max_tokens.value if hasattr(max_tokens, 'value') else 512)
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': reply})
        return '', history

    def user_submit_with_params(message, image, history, temp, max_tok):
        reply = respond(message, image, history, temp, int(max_tok))
        history.append({'role': 'user', 'content': message})
        history.append({'role': 'assistant', 'content': reply})
        return '', history

    msg.submit(user_submit_with_params, [msg, image_input, chatbot, temperature, max_tokens], [msg, chatbot])
    clear_btn.click(lambda: (None, [], ''), outputs=[image_input, chatbot, msg])

demo.launch(server_name='0.0.0.0', server_port=port, share=share)
"
