import streamlit as st

st.set_page_config(
        page_title="FastLegal - Supercharge your Legal Research",
        page_icon="🔍",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# Find me at adrien@stepone.agency"
            }
    )

st.write("# Welcome to FastLegal! 👋")

st.sidebar.success("Select a search tool above.")

st.markdown(
    """
    FastLegal is the ultimate in legal research, powered by advanced ChatGPT-4 technology. This AI-powered tool enables you to search the Hong Kong legal database instantly and accurately, providing you with the insights you need to make informed decisions. 
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [fastlegal.io](https://harryhtkwong.wixsite.com/fastlegal)
    - Schedule a [demo](https://harryhtkwong.wixsite.com/fastlegal)
    - Reach out for [partnership opportunities](https://harryhtkwong.wixsite.com/fastlegal)
    ### Get Started
    - Start by selecting a search tool on the left side bar.
"""
)