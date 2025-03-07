import streamlit as st
from helper import get_QA_chain

chain = get_QA_chain()

def get_response(question):
    chain = get_QA_chain()
    ans = chain.invoke({"input": question})
    return ans
    # print(ans)

def get_relevant_prompts(chain,ans):     
    st.write('\nResponse from Model: '.upper())
    st.write(ans['answer'])

    st.write('\n# [References From Data]:\n'.upper())
    num_of_indexs = len(ans['context'])
    for i in range(num_of_indexs):
        st.write(i+1,((ans['context'])[i]).page_content)
        if i == 4:
            break
       

st.title("The Bridge of Hopes Chatbot Q&A ")
# btn = st.button("Create Knowledgebase")
# if btn:
#     create_vector_db(folder_path)

question = st.text_input("Question: ")

if question:
    response = get_response(question)
    st.header("Answer")
    # st.write(response["answer"])
    get_relevant_prompts(chain,response)
    print(chain)
