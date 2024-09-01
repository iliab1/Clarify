import streamlit as st
import os
import requests

def validate_openai_api_key(api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            return True, "OpenAI API key is valid."
        else:
            return False, f"OpenAI API key validation failed: {response.json().get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return False, f"OpenAI API key validation failed: {str(e)}"

def validate_google_api_key(api_key, cse_id):
    try:
        params = {
            "key": api_key,
            "cx": cse_id,
            "q": "test"
        }
        response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
        if response.status_code == 200:
            return True, "Google API key and CSE ID are valid."
        else:
            error_message = response.json().get('error', {}).get('message', 'Unknown error')
            # Check specific error messages
            if "Request contains an invalid argument" in error_message:
                return False, "Google API validation failed: Invalid CSE ID. Please provide a correct Google CSE ID."
            elif "API key not valid" in error_message:
                return False, "Google API validation failed: API key not valid. Please provide a valid API key."
            else:
                return False, f"Google API validation failed: {error_message}"
    except Exception as e:
        return False, f"Google API validation failed: {str(e)}"

def sidebar_settings():
    st.sidebar.title('Settings')

    # Section for API key management
    st.sidebar.subheader("API Key Management")

    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    google_api_key = st.sidebar.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id = st.sidebar.text_input("Google CSE ID", type="password", value=os.getenv("GOOGLE_CSE_ID", ""))

    if st.sidebar.button("Update API Keys"):
        # Validate OpenAI API Key

        is_openai_key_valid, openai_message = validate_openai_api_key(openai_api_key)
        if is_openai_key_valid:
            # st.sidebar.success(openai_message)
            pass
        else:
            st.sidebar.error(openai_message)
            st.stop()

        # Validate Google API Key and CSE ID
        is_google_key_valid, google_message = validate_google_api_key(google_api_key, google_cse_id)
        if is_google_key_valid:
            # st.sidebar.success(google_message)
            pass
        else:
            st.sidebar.error(google_message)
            st.stop()

        st.sidebar.success("API keys have been updated successfully.")