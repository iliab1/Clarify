import streamlit as st
from chains import full_chain
from langchain_openai import ChatOpenAI

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from sidebar_settings import sidebar_settings
sidebar_settings()

# Custom CSS
st.markdown("""
    <style>
        .title {
            color: #1F77B4; /* Blue color for title */
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🔍 Clarify: Claim Verification System</h1>", unsafe_allow_html=True)



llm_openai = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

question = st.text_input('Enter a claim to verify:', 'The earth is flat')

def get_colour_code(verdict):
    if verdict == "True":
        colour = "🟢"
    elif verdict == "False":
        colour = "🔴"
    else:
        colour = "🟡"
    return colour

st.write("")
if question and st.button('Verify'):
    with st.spinner('Verifying...'):
        # Invoke the chain
        results = full_chain(llm_openai).invoke({"text": question})
        st.write(results)
        # Display the results in expandable containers
        for idx, claim in enumerate(results["fact_checked_claims"]):
            verdict = claim['answer']['verdict']
            colour_code = get_colour_code(verdict)
            title = f"Claim {idx + 1}: {claim['input']}"
            sources = claim['answer']['sources']
            with st.expander(title, icon=colour_code):
                st.write(f"**Claim:** {claim['input']}")

                st.write(f"**Verdict:** {colour_code} {claim['answer']['verdict']}")
                st.write(f"**Answer:**")
                st.markdown(f"{claim['answer']['answer']}")
                st.write("**Sources:**")
                if sources:
                    for source in sources:
                        st.markdown(f"- {source}")
                else:
                    st.write("None")








