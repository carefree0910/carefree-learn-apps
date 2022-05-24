import cflearn_apps
import streamlit as st


url = "carefree-learn-deploy:8000"
client = cflearn_apps.Client(url=url)

app = cflearn_apps.MultiPage(client)
st.image(
    "https://socialify.git.ci/carefree0910/carefree-learn/image?"
    "description=1&descriptionEditable=Deep%20Learning%20%E2%9D%A4%EF%B8%8F%20"
    "PyTorch&forks=1&issues=1&logo=https%3A%2F%2Fraw.githubusercontent.com"
    "%2Fcarefree0910%2Fcarefree-learn-doc%2Fmaster%2Fstatic%2Fimg%2Flogo.min.svg"
    "&pattern=Floating%20Cogs&stargazers=1"
)

st.sidebar.title(
    "Demos for [carefree-learn]"
    "(https://github.com/carefree0910/carefree-learn/tree/v0.2.x)"
)

app.add_page("Salient Object Detection", cflearn_apps.sod.app)
app.run()
