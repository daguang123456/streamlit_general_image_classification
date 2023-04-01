import streamlit as st
from PIL import Image
from clf import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("使用 Resnet101 进行常规图像分类")

st.write("教程在这里 [link](https://github.com/dehaoterryzhang/Image_Classification_App)")

file_up = st.file_uploader("上传图片", type=["jpg","jpeg","png"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("等一下...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("预测 (index, name)", i[0], i[1],",   Score: ", i[2])
