FROM python:3.10

WORKDIR /chat_ui

COPY requirements.txt /chat_ui/requirements.txt
COPY .env /chat_ui/.env

RUN pip install --no-cache-dir --upgrade -r /chat_ui/requirements.txt

EXPOSE 7860

COPY langchain_gradio_chat_stream.py /chat_ui/langchain_gradio_chat_stream.py

CMD ["python", "langchain_gradio_chat_stream.py"]