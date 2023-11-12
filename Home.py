import streamlit as st


def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: url('https://i.imgur.com/XlV61vK.png');
                background-repeat: no-repeat;
                padding-top: 30px;
                background-position: 20px 50px;
                background-size: 300px 50px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )



st.set_page_config(
        page_title="FastLegal - Supercharge your Legal Research",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# Find me at adrien@stepone.agency"
            }
    )

add_logo()

# st.sidebar.image("assets/FastLegal Logo - transparent_rec.png", width=200)
# st.image("assets/FastLegal Logo - transparent_rec.png", width=300)
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

