
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAIChat
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.docstore.document import Document
from langchain.callbacks import get_openai_callback


@st.cache_resource
def getYoutubeScript(videoid):
    #st.write("Not in New cache")
    transcript_list = YouTubeTranscriptApi.get_transcript(videoid)
    formatter = TextFormatter()
    formatted = formatter.format_transcript(transcript_list)
    return formatted

video_id = ""
fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give detailed sentences. Show atleast 2 sentences :\n\n {text_input}"
)


st.title("🧠 Chat with your Youtube Videos 🤖")

st.sidebar.markdown(
    """
    ### Steps:
    1. Enter Youtube URL link
    2. Enter Your Secret Key from OpenAI
    3. Get You tube Video summaries in diff formats. 
**Note : File content and API key not stored in any form.**
    """
)

api = st.sidebar.text_input(
            "**Enter OpenAI API Key**",
            type="password",
            placeholder="sk-",
            help="https://platform.openai.com/account/api-keys",
        )

prompt = PromptTemplate(
    input_variables=["docs"],
    template="Please summarise the following text in two short bullet points. \n {docs}"
)
if api:
    llm = OpenAI(temperature=0.1,openai_api_key=api, model_name="text-davinci-003",max_tokens=512)

#st.image("https://seeklogo.com/images/Y/youtube-icon-logo-521820CDD7-seeklogo.com.png")
video_url = st.text_input("Enter your Youtube URL: ")

output_size = st.radio(label = "What kind of output do you want?", 
                    options= ["TLDR", "Concise", "Bullets","Verbose"])

chunksize = 2000

def SummariseLLM(doc):   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    #print(len(docs))
    chain = load_summarize_chain(llm,chain_type="map_reduce",verbose=False)
    #cchain = SimpleSequentialChain(chains=[chain],llm=llm, prompt=prompt)
    res = chain.run(docs)
    return res

def SummariseLLMBullets(result):   
    chunksize = 2000
    
    num_iters = int(len(result)/chunksize)
    summarized_text = []
    res = []
    summ = []
    for i in range(0, num_iters + 1):
        start = 0
        start = i * chunksize
        end = (i + 1) * chunksize
        #print("input text \n" + result[start:end])
        doc = [Document(page_content=result[start:end])]
        #print(len(docs))
        chain = LLMChain(prompt=prompt,llm=llm)
        #cchain = SimpleSequentialChain(chains=[chain],llm=llm, prompt=prompt)
        res = chain.run(doc)
        summ.append(res)
    return summ

def SummariseLLMRefine(doc):   
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(doc)
    #print(len(docs))
    chain = load_summarize_chain(llm,chain_type="refine",verbose=False)
    #cchain = SimpleSequentialChain(chains=[chain],llm=llm, prompt=prompt)
    res = chain.run(docs)
    return res

def writeVideoFile(video_id, transcript):
    doc = [Document(page_content=transcript)]
    #with open( video_id + ".txt", 'w') as f:
        #f.write(transcript)
    #loader = DirectoryLoader(path=".", glob="**/" + video_id + ".txt")
    res = SummariseLLM(doc)
    return res

def getChunks(result):
    chunksize = 2000
    #llm = OpenAI(temperature=0.1)
    num_iters = int(len(result)/chunksize)
    summarized_text = []
    res = []
    summ = []
    for i in range(0, num_iters + 1):
        start = 0
        start = i * chunksize
        end = (i + 1) * chunksize
        #print("input text \n" + result[start:end])
        doc = [Document(page_content=result[start:end])]
        chain = load_summarize_chain(llm,chain_type="map_reduce",verbose=False)
        #fact_extraction_prompt.format(text_input= docs)
        #chain = LLMChain(llm=llm, prompt=fact_extraction_prompt)
        res = chain.run(doc)
        #out = summarizer(result[start:end])
        #out = out[0]
        #out = out['summary_text']
        summ.append(res + '\n')
    return summ
        #summarized_text.append(out)
        #      
if st.button("Generate Summary",type='primary') and api:
    with st.spinner(text="In progress..."):
        strtext=""
        if (video_id == ""):
            video_id = video_url.split("=")[1]
            formatted_text = getYoutubeScript(video_id)
            doc = [Document(page_content=t) for t in formatted_text]
            #llm = OpenAI(temperature=0.1)
            #fact_extraction_chain = LLMChain(llm=llm,prompt=fact_extraction_prompt)
            fact_extraction_chain = load_summarize_chain(llm=llm,chain_type="map_reduce",verbose=False)
            mapredsum = writeVideoFile(video_id,formatted_text)   
        
        if (output_size == "Verbose"):
            if len(formatted_text) > 50:
                res = formatted_text
                st.success(res)
                st.download_button('Download result', res)
        elif (output_size=="TLDR" ):
            if len(formatted_text) > 50:
                #res = writeVideoFile(video_id, formatted_text)
                #res = SummariseLLM(formatted_text)
                st.success(mapredsum)
                st.download_button('Download result', mapredsum)
        elif (output_size=="Bullets"):
            if len(formatted_text) > 50:
                res=[]
                #res = writeVideoFile(video_id, formatted_text)
                res = SummariseLLMBullets(formatted_text)
                summtext = ""
                for  r in res:
                    summtext = summtext + r + '\n'
                st.success(summtext)
                st.download_button('Download result', summtext)
        else:
                summtext = ""
                summ = getChunks(formatted_text)
                #strtext = ' '.join([strtext for t in summ])
                con = []
                for txt in summ:
                    summtext = summtext + txt + '\n'
                st.success(summtext)
                st.download_button('Download result', summtext)
