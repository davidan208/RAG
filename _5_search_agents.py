from langchain_community.tools import DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from _3_llm_generate import get_hf_llm

class SearchAgent:
    def __init__(self):
        self.search_tool = DuckDuckGoSearchResults()
        self.llm = get_hf_llm(temperature = 0)

        self.search_prompt = PromptTemplate(
            input_variables= ["query", "search_result"],
            template = """
            Với kết quả tìm kiếm sau, hãy cung cấp câu trả lời ngắn gọn và đầy đủ thông tin cho truy vấn.
            Nếu kết quả tìm kiếm không chứa thông tin có liên quan, hãy nêu rằng thông tin đó không có sẵn.

            Câu hỏi: {query}
            Tìm kiếm: {search_result}
            Trả lời: 
            """
        )

        self.search_chain = LLMChain(llm = self.llm, prompt = self.search_prompt)
    
    def search(self, query: str) -> str:
        search_result = self.search_tool.run(query)
        output_response = self.search_chain.run(query = query, search_result = search_result)
        return self.format_response(output_response)

    def format_response(self, response: str) -> str:
        formatted_response = "[LƯU Ý]: Kết quả có thể không chính xác do được thực hiện dựa trên việc tìm kiếm qua internet \n\n"
        formatted_response += f"{response}\n"
        return formatted_response
    






