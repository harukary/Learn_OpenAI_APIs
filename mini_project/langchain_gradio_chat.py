import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0613", 
    max_tokens=512,
    temperature=0.2,
)
conversation = ConversationChain(llm=llm, verbose=False)

chat_history = []
def chat(message):
    response = conversation.predict(input=message)
    print(response)
    return response

css = """
.gradio-container {background-color: #7494C0}
.message.user{
    background: #8DE055 !important;
}
.message.assistant{
    background: #EDF1EE !important;
}
"""

with gr.Blocks(css=css) as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def user(message, chat_history):
        return "", chat_history + [[message, None]]
    
    def respond(chat_history):
        response = chat(chat_history[-1][0])
        chat_history[-1][1] = response
        return chat_history

    # msg.submit(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        respond, chatbot, chatbot
    )

if __name__ == "__main__":
    demo.launch(debug=True,)