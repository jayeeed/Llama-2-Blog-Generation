from torch import cuda
import transformers
from langchain_community.llms import HuggingFacePipeline

model_id = 'meta-llama/Llama-2-7b-chat-hf'
hf_auth = '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'

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    device_map='auto',
    token=hf_auth
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    temperature=0.5,
    max_new_tokens=256,
    repetition_penalty=1.5
)

llm = HuggingFacePipeline(pipeline=generate_text)

def main():
    while True:
        prompt = input("Enter your prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        res = llm(prompt=prompt)
        
        generated_text = res
        
        # Find the index of "Answer:"
        answer_index = generated_text.find("Answer:")
        
        if answer_index != -1:  # If "Answer:" is found
            # Extract the text that follows "Answer:"
            generated_text = generated_text[answer_index + len("Answer:"):]
            # Remove leading and trailing whitespace
            generated_text = generated_text.strip()
        
        print("\nGenerated Text:")
        print(generated_text)
        print("\n")

if __name__ == "__main__":
    main()
