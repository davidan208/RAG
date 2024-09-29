from _3_llm_generate import get_hf_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence

import os
import glob
import json

json_dir = './json_src'
if not os.path.exists(json_dir):
    raise FileExistsError("[CRAWL ERROR] Không tìm thấy file json sau khi crawl")

json_file = glob.glob(json_dir + "/*.json")[0]  # Only one json file is allowed to be existed

with open('test.json', 'r') as file:
    data = json.load(file)

content = data['content']


# HuggingFace -> OpenAI
llm = get_hf_llm(temperature = 0)

related_prompt_template = '''
content: {content}
Sinh ra 20 câu hỏi hoàn toàn khác nhau chỉ dựa trên content.
'''

related_prompt = PromptTemplate(
    input_variables=["content"],
    template=related_prompt_template,
)

related_chain = RunnableSequence(first = related_prompt, last = llm)

related_questions = related_chain.invoke({
    "content": content
})