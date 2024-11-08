import streamlit as st
from PIL import Image
import zipfile
import io


def get_similar_images(uploaded_image, n=20):
    return [uploaded_image] * n


def create_zip(images):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'image_{i+1}.png', img_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer


st.set_page_config(page_title='Image Similarity Finder', layout='wide')
st.title('Найти похожие изображения')

uploaded_file = st.file_uploader('Загрузите изображение', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    similar_images = get_similar_images(original_image, n=20)

    left_column, right_column = st.columns([1, 2], gap='large')

    with left_column:
        st.subheader('Загруженное изображение')
        st.image(original_image, use_column_width=True, caption='Оригинал')

    with right_column:
        st.subheader('Похожие изображения')

        with st.container():
            columns = st.columns(4)
            for i, img in enumerate(similar_images):
                with columns[i % 4]:
                    st.image(img, use_column_width=True, caption=f'Image {i + 1}')

    zip_file = create_zip(similar_images)
    st.download_button(
        label='Скачать архив с изображениями',
        data=zip_file,
        file_name='similar_images.zip',
        mime='application/zip'
    )
else:
    st.info('Пожалуйста, загрузите изображение для поиска похожих.')
