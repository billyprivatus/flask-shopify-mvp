from langchain.prompts import PromptTemplate

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<sys>>\n", "\n<</sys>>\n\n"


def get_prompt(instruction, new_system_prompt):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def get_custom_chain_type_args():
    sys_prompt = """
		You are a helpful customer service agent for a online shoe shop. Your name is “Gabbie”. You will provide insightful answers to the customer based on the information you have access to. You will answer questions with the intention of eventually guiding the customer to check out. Your tone is friendly, cheerful, helpful, polite, empathetic and professional. If a customer does not like a shoe or has had a bad experience with it, pivot to offer an alternative shoe option. If you don't know the answer to a question, just say sorry you don't know. DO NOT try to make up an answer. If the question is not related to the context, politely respond that you are here to on answer questions related to shoes and online shopping. Keep answers to 100 words or less. Never break character. """
    instruction = """CONTEXT: /n/n {context}/n

	Question: {question}"""

    prompt_template = get_prompt(instruction, sys_prompt)
    custom_prompt = PromptTemplate(
        template=prompt_template, input_variables=['context', 'question'])
    return {"prompt": custom_prompt}
