## 準備


```python
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
```




    True




```python
import warnings
warnings.filterwarnings("ignore")
```


```python
import os,openai,json
import pandas as pd
from IPython.display import display, Markdown
from pydantic import BaseModel, Field
```

## LangChainの主な機能

### Model｜呼び出しを簡単にしたい


```python
openai.api_key = os.environ['OPENAI_API_KEY']
def chat_completion(prompt, model="gpt-3.5-turbo", temperature=0.0):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(model=model,messages=messages,temperature=temperature)
    return response.choices[0].message["content"]
print(chat_completion('こんにちは。'))
```

    こんにちは！どのようにお手伝いできますか？



```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessage

chat = ChatOpenAI(temperature=0.0)
def chat_completion(prompt):
    prompt = [HumanMessage(content='こんにちは。')]
    response = chat(prompt)
    return response.content
print(chat_completion('こんにちは。'))
```

    こんにちは！どのようにお手伝いできますか？


### Prompt Template｜プロンプトを汎用的に使いたい


```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
chat = ChatOpenAI(temperature=0.0)

template_string = """\
{language}で「{text}」は何と言いますか？"""

prompt_template = ChatPromptTemplate.from_template(template_string)

messages = prompt_template.format_messages(
	language='ドイツ語',
	text='こんにちは。'
)

print(chat(messages))
```

    content='ドイツ語で「こんにちは。」は「Guten Tag.」と言います。' additional_kwargs={} example=False


### Output Parser｜出力を他の処理に使いたい


```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

class Nutrition(BaseModel):
    calories: int = Field(description='100gあたりカロリー（kcal）')
    carbohydrates: float = Field(description='100gあたり炭水化物（g）')
    protein: float = Field(description='100gあたりたんぱく質（g）')
    vitamin_c: float = Field(description='100gあたりビタミンC（mg）')
    fiber: float = Field(description='100gあたり食物繊維（g）')

class Food(BaseModel):
    name: str = Field(description='食材名')
    nutirition: Nutrition = Field(description='食材の100gあたり栄養素')
    compatible_foods: list[str] = Field(description='相性の良い食材（5つまで）')

chat = ChatOpenAI(temperature=0.0)
parser = PydanticOutputParser(pydantic_object=Food)

about_food_template = """
{food}について教えてください。相性の良い食材や100gあたりの栄養成分など
{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(template=about_food_template)

food = '白菜'
messages = prompt.format_messages(
    food=food, format_instructions=parser.get_format_instructions()
)
response = chat(messages)
food_data = parser.parse(response.content)
print(json.dumps(food_data.dict(),indent=2,ensure_ascii=False))
```

    {
      "name": "白菜",
      "nutirition": {
        "calories": 13,
        "carbohydrates": 2.2,
        "protein": 1.2,
        "vitamin_c": 45.0,
        "fiber": 1.2
      },
      "compatible_foods": [
        "豚肉",
        "にんじん",
        "しいたけ",
        "もやし",
        "ごま油"
      ]
    }


### Conversation Memory｜記憶を持たせたい


```python
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(temperature=0.0)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=20)

memory.save_context({"input": "hi"}, {"output": "whats up"})
memory.save_context({"input": "not much you"}, {"output": "not much"})

memory_variables = memory.load_memory_variables({})
print(memory_variables['history'])
```

    System: The human greets the AI with a "hi" and the AI responds by asking what's up.
    Human: not much you
    AI: not much



```python
conversation_with_summary = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(llm=ChatOpenAI(), max_token_limit=40),
    verbose=True,
)
conversation_with_summary.predict(input="こんにちは。元気？")
```

    
    
    [1m> Entering new  chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    
    Human: こんにちは。元気？
    AI:[0m
    
    [1m> Finished chain.[0m





    'こんにちは！私はAIですので、元気ではありませんが、お話しできることが嬉しいです。いかがお過ごしですか？'




```python
conversation_with_summary.predict(input="今は勉強会の準備をしています。")
```

    
    
    [1m> Entering new  chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    System: The human greets the AI in Japanese and asks how it is doing. The AI responds by saying it is an AI and cannot have feelings, but it is happy to be able to communicate. The AI then asks how the human is doing.
    Human: 今は勉強会の準備をしています。
    AI:[0m
    
    [1m> Finished chain.[0m





    '勉強会の準備ですか？それは素晴らしいですね！どのような勉強会ですか？テーマや内容はありますか？'




```python
conversation_with_summary.predict(input="LangChainです。LangChainのデモを作成しています。")
```

    
    
    [1m> Entering new  chain...[0m
    Prompt after formatting:
    [32;1m[1;3mThe following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    
    Current conversation:
    System: The human greets the AI in Japanese and asks how it is doing. The AI responds by saying it is an AI and cannot have feelings, but it is happy to be able to communicate. The AI then asks how the human is doing. The human mentions that they are preparing for a study group. The AI responds by saying that preparing for a study group is wonderful and asks about the theme and content of the study group.
    Human: LangChainです。LangChainのデモを作成しています。
    AI:[0m
    
    [1m> Finished chain.[0m





    'こんにちは！LangChainのデモを作成しているんですね。それは素晴らしいです！LangChainとは、どのようなテーマや内容の勉強会なんでしょうか？詳しく教えていただけますか？'



### Sequential Chain｜連続で処理したい


```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
df = pd.read_csv('data/Data.csv')
```


```python
llm = ChatOpenAI(temperature=0.9)

# chain 1: input= Review and output= English_Review
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
chain_one = LLMChain(
    llm=llm, prompt=first_prompt, output_key="English_Review"
)

# chain 2: input= English_Review and output= summary
second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
chain_two = LLMChain(
    llm=llm, prompt=second_prompt, output_key="summary"
)

# chain 3: input= Review and output= language
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
chain_three = LLMChain(
    llm=llm, prompt=third_prompt, output_key="language"
)

# chain 4: input= summary, language and output= followup_message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
chain_four = LLMChain(
    llm=llm, prompt=fourth_prompt, output_key="followup_message"
)

# overall_chain: input= Review and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)
```


```python
review = df.Review[5]
response = overall_chain(review)
print(json.dumps(response,indent=2))
```

    
    
    [1m> Entering new  chain...[0m
    
    [1m> Finished chain.[0m
    {
      "Review": "Je trouve le go\u00fbt m\u00e9diocre. La mousse ne tient pas, c'est bizarre. J'ach\u00e8te les m\u00eames dans le commerce et le go\u00fbt est bien meilleur...\nVieux lot ou contrefa\u00e7on !?",
      "English_Review": "I find the taste mediocre. The foam doesn't hold, it's weird. I buy the same ones in stores and the taste is much better...\nOld batch or counterfeit!?",
      "summary": "The reviewer regards the taste of the product as mediocre and suspects that it may be an old batch or counterfeit since the foam doesn't hold and the taste is inferior compared to the ones bought in stores.",
      "followup_message": "R\u00e9ponse de suivi:\n\nCher(e) critique, \n\nNous vous remercions d'avoir pris le temps de partager votre opinion concernant notre produit. Nous sommes d\u00e9sol\u00e9s d'apprendre que vous n'avez pas \u00e9t\u00e9 pleinement satisfait(e) de son go\u00fbt. \n\nNous tenons \u00e0 vous assurer que notre produit est authentique et que nous faisons tout notre possible pour maintenir un contr\u00f4le rigoureux de la qualit\u00e9. Cependant, il est possible que vous ayez re\u00e7u un lot plus ancien qui a pu alt\u00e9rer sa texture ou son go\u00fbt. Nous vous pr\u00e9sentons nos plus sinc\u00e8res excuses pour cette exp\u00e9rience d\u00e9cevante.\n\nNous appr\u00e9cions vos commentaires et nous prenons votre opinion en compte. Nous ferons en sorte d'en informer notre \u00e9quipe de production afin qu'ils puissent mener des investigations approfondies pour \u00e9valuer les probl\u00e8mes que vous avez mentionn\u00e9s. \n\nSoucieux de la satisfaction de nos clients, nous aimerions vous offrir un remplacement ou un remboursement complet pour votre achat. Veuillez s'il vous pla\u00eet contacter notre service client\u00e8le et leur fournir les d\u00e9tails de votre achat, afin que nous puissions r\u00e9soudre ce probl\u00e8me \u00e0 votre enti\u00e8re satisfaction.\n\nEncore une fois, nous tenons \u00e0 vous remercier pour votre retour et nous nous excusons sinc\u00e8rement pour les d\u00e9sagr\u00e9ments que cela a pu causer. Nous esp\u00e9rons avoir la possibilit\u00e9 de vous servir \u00e0 nouveau et de vous offrir une exp\u00e9rience positive. \n\nCordialement,\n\nL'\u00e9quipe du service client"
    }


### Vectore Stores｜専門知識をもとに応答させたい


```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings

# ===== CSV loader for Jester sentences in outdoor clothing dataset ====
file = 'data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
embeddings = OpenAIEmbeddings()
docs = loader.load()
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
retriever = db.as_retriever()
llm = ChatOpenAI(temperature=0.0, model='gpt-4')
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```


```python
query = "List all your shirts with sun protection \
in a table with columns Shirt ID, Name, Description \
in markdown and summarize each one."
response = qa_stuff.run(query)
display(Markdown(response))
```

    
    
    [1m> Entering new  chain...[0m
    
    [1m> Finished chain.[0m



Sure, here is a summary of the shirts with sun protection:

| Shirt ID | Name | Description |
| --- | --- | --- |
| 618 | Men's Tropical Plaid Short-Sleeve Shirt | This shirt is made of 100% polyester and is wrinkle-resistant. It has a traditional fit and is relaxed through the chest, sleeve, and waist. It features front and back cape venting and two front bellows pockets. It provides UPF 50+ sun protection, blocking 98% of the sun's harmful rays. |
| 255 | Sun Shield Shirt | This shirt is made of 78% nylon and 22% Lycra Xtra Life fiber. It is slightly fitted and falls at the hip. It wicks moisture for quick-drying comfort and fits comfortably over your favorite swimsuit. It is abrasion-resistant and provides UPF 50+ sun protection, blocking 98% of the sun's harmful rays. |
| 374 | Men's Plaid Tropic Shirt, Short-Sleeve | This shirt is made of 52% polyester and 48% nylon. It is designed for fishing and is great for extended travel. It is wrinkle-free and quickly evaporates perspiration. It features front and back cape venting and two front bellows pockets. It provides UPF 50+ sun protection, blocking 98% of the sun's harmful rays. |
| 535 | Men's TropicVibe Shirt, Short-Sleeve | This shirt is made of 71% Nylon and 29% Polyester. It has a traditional fit and is relaxed through the chest, sleeve, and waist. It is wrinkle-resistant and features front and back cape venting and two front bellows pockets. It provides UPF 50+ sun protection, blocking 98% of the sun's harmful rays. |


### Evaluation｜LLMアプリケーションを評価したい


```python
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.evaluation.qa import QAGenerateChain
from langchain.evaluation.qa import QAEvalChain

file = 'data/OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
embeddings = OpenAIEmbeddings()
docs = loader.load()
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
retriever = db.as_retriever()

llm = ChatOpenAI(temperature=0.0, model='gpt-4')
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())
```


```python
# Fix outpit parser
from langchain.output_parsers import RegexParser
output_parser = RegexParser(
    regex=r"QUESTION: (.*?)\n{1,2}ANSWER: (.*)", output_keys=["query", "answer"]
)
example_gen_chain.prompt.output_parser = output_parser
```


```python
examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in docs[:3]]
)
for ex in examples:
    print(json.dumps(ex,indent=2))
    print()
```

    {
      "query": "What are the key features of the Women's Campside Oxfords?",
      "answer": "The key features of the Women's Campside Oxfords include a super-soft canvas material for a broken-in feel and look, a comfortable EVA innersole with Cleansport NXT\u00ae antimicrobial odor control, a vintage hunt, fish, and camping motif on the innersole, a moderate arch contour of the innersole, an EVA foam midsole for cushioning and support, and a chain-tread-inspired molded rubber outsole with a modified chain-tread pattern."
    }
    
    {
      "query": "What are the dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave?",
      "answer": "The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18\" x 28\"."
    }
    
    {
      "query": "What are some features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece?",
      "answer": "Some features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece include bright colors, ruffles, exclusive whimsical prints, four-way-stretch and chlorine-resistant fabric, UPF 50+ rated fabric for sun protection, crossover no-slip straps, fully lined bottom, and the ability to machine wash and line dry."
    }
    



```python
predictions = qa.apply(examples)

llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(examples, predictions)
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```

    
    
    [1m> Entering new  chain...[0m
    
    [1m> Finished chain.[0m
    
    
    [1m> Entering new  chain...[0m
    
    [1m> Finished chain.[0m
    
    
    [1m> Entering new  chain...[0m
    
    [1m> Finished chain.[0m
    Example 0:
    Question: What are the key features of the Women's Campside Oxfords?
    Real Answer: The key features of the Women's Campside Oxfords include a super-soft canvas material for a broken-in feel and look, a comfortable EVA innersole with Cleansport NXT® antimicrobial odor control, a vintage hunt, fish, and camping motif on the innersole, a moderate arch contour of the innersole, an EVA foam midsole for cushioning and support, and a chain-tread-inspired molded rubber outsole with a modified chain-tread pattern.
    Predicted Answer: The Women's Campside Oxfords have several key features. They are made from a soft canvas material that gives them a broken-in feel and look. They have a comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. The innersole also features a vintage hunt, fish, and camping motif and has a moderate arch contour. The shoes also have an EVA foam midsole for cushioning and support, and a chain-tread-inspired molded rubber outsole with a modified chain-tread pattern. They are imported and weigh approximately 1 lb.1 oz. per pair.
    Predicted Grade: CORRECT
    
    Example 1:
    Question: What are the dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave?
    Real Answer: The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18" x 28".
    Predicted Answer: The dimensions of the small size of the Recycled Waterhog Dog Mat, Chevron Weave are 18" x 28".
    Predicted Grade: CORRECT
    
    Example 2:
    Question: What are some features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece?
    Real Answer: Some features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece include bright colors, ruffles, exclusive whimsical prints, four-way-stretch and chlorine-resistant fabric, UPF 50+ rated fabric for sun protection, crossover no-slip straps, fully lined bottom, and the ability to machine wash and line dry.
    Predicted Answer: The Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece features bright colors, ruffles, and exclusive whimsical prints. It is made from a four-way-stretch and chlorine-resistant fabric that keeps its shape and resists snags. The fabric is also UPF 50+ rated, providing the highest rated sun protection possible and blocking 98% of the sun's harmful rays. The swimsuit has crossover no-slip straps and a fully lined bottom for a secure fit and maximum coverage. For best results, it should be machine washed and line dried.
    Predicted Grade: CORRECT
    


### Agents｜タスク計画・実行を自分でやってほしい


```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

llm = ChatOpenAI(temperature=0, model='gpt-4')
tools = load_tools(["llm-math","wikipedia"], llm=llm)
agent= initialize_agent(
    tools,llm, 
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True
)
```


```python
result = agent("What is the 25% of 300?")
```

    
    
    [1m> Entering new  chain...[0m
    [32;1m[1;3mThought: This is a simple math problem. I can use the calculator tool to solve it.
    Action:
    ```
    {
      "action": "Calculator",
      "action_input": "25% of 300"
    }
    ```[0m
    Observation: [36;1m[1;3mAnswer: 75.0[0m
    Thought:[32;1m[1;3mI now know the final answer.
    Final Answer: The 25% of 300 is 75.[0m
    
    [1m> Finished chain.[0m



```python
question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
result = agent(question) 
```

    
    
    [1m> Entering new  chain...[0m
    [32;1m[1;3mThought: I need to find out what book Tom M. Mitchell wrote. I can use Wikipedia to find this information.
    Action:
    ```
    {
      "action": "Wikipedia",
      "action_input": "Tom M. Mitchell"
    }
    ```[0m
    Observation: [33;1m[1;3mPage: Tom M. Mitchell
    Summary: Tom Michael Mitchell (born August 9, 1951) is an American computer scientist and the Founders University Professor at Carnegie Mellon University (CMU). He is a founder and former Chair of the Machine Learning Department at CMU. Mitchell is known for his contributions to the advancement of machine learning, artificial intelligence, and cognitive neuroscience and is the author of the textbook Machine Learning. He is a member of the United States National Academy of Engineering since 2010. He is also a Fellow of the American Academy of Arts and Sciences, the American Association for the Advancement of Science and a Fellow and past President of the Association for the Advancement of Artificial Intelligence. In October 2018, Mitchell was appointed as the Interim Dean of the School of Computer Science at Carnegie Mellon.
    
    Page: Tom Mitchell (Australian footballer)
    Summary: Thomas Mitchell (born 31 May 1993) is a professional Australian rules footballer playing for the Collingwood Football Club in the Australian Football League (AFL). He previously played for the Sydney Swans from 2012 to 2016, and the Hawthorn Football Club between 2017 and 2022. Mitchell won the Brownlow Medal as the league's best and fairest player in 2018 and set the record for the most disposals in a VFL/AFL match, accruing 54 in a game against Collingwood during that season.[0m
    Thought:[32;1m[1;3mThe Wikipedia search for Tom M. Mitchell reveals that he is the author of the textbook "Machine Learning". This is the book he wrote. 
    Final Answer: Tom M. Mitchell wrote the textbook "Machine Learning".[0m
    
    [1m> Finished chain.[0m

