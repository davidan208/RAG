from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig
import os
import torch
from langchain_huggingface import HuggingFacePipeline

def get_hf_llm( model : str = "Viet-Mistral/Vistral-7B-Chat",
                            max_new_tokens = 1024,
                            repetition_penalty = 1.19,
                            **kwargs):
    offline_models = './offline_models'

    if not os.path.isdir(offline_models):
        os.mkdir(offline_models)

    # Viet-Mistral
    link_to_Vistral = model
    model       = AutoModelForCausalLM.from_pretrained( link_to_Vistral,
                                                        cache_dir = offline_models,
                                                        quantization_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_quant_type = "nf4", bnb_4bit_use_double_quant = True, bnb_4bit_compute_dtype = torch.bfloat16),
                                                        low_cpu_mem_usage = True,
                                                        torch_dtype = torch.bfloat16,
                                                        device_map = "auto"
                                                        )
    
    tokenizer   = AutoTokenizer.from_pretrained(link_to_Vistral, device_map = "auto")

    model_pipeline = pipeline(  'text-generation',
                                model = model,
                                tokenizer = tokenizer,
                                max_new_tokens = max_new_tokens,
                                pad_token_id = tokenizer.eos_token_id,
                                device_map = "auto",
                                early_stopping = True,
                                repetition_penalty = repetition_penalty
                                )
    
    llm = HuggingFacePipeline(pipeline = model_pipeline, model_kwargs= kwargs)

    return llm