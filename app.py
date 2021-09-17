import cflearn_apps
import streamlit as st


app = cflearn_apps.MultiPage()
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

app.add_page("Style GAN", cflearn_apps.style_gan.app)
app.add_page("Color Extraction", cflearn_apps.color_extraction.app)
app.add_page("Salient Object Detection", cflearn_apps.sod.app)
app.add_page("Content Based Image Retrieval", cflearn_apps.cbir.app)
app.add_page("Text Based Image Retrieval", cflearn_apps.tbir.app)
app.add_page("Arbitrary Style Transfer with AdaIN", cflearn_apps.adain.app)
app.add_page("Image Classification", cflearn_apps.clf.app)
app.run()
