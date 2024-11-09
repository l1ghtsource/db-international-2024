import streamlit as st
import zipfile
from faiss import read_index
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from faiss import read_index
import pickle
import cv2
from ultralytics import YOLO
import io
from get_mapping import get_mapping

# get the class mapping dictionary
class_mapping_dict = get_mapping()

# load the YOLO model
yolo = YOLO('yolov8x-oiv7.pt')

# function to find similar images using CLIP
def find_similar_images_clip(model, faiss_index, image_paths, query_image, top_k=10):
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # prepare the image for the CLIP model
    inputs = processor(images=query_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # ensure the model is on the correct device
    model.to(device)

    # Extract features
    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding.cpu().numpy()[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
    
    # search in the Faiss index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # rerturn the similar images with their paths
    similar_images = [image_paths[i] for i in indices[0]]
    
    # extract class names (parent folder of each image)
    class_names = [image_path.split('/')[-2] for image_path in similar_images]
    
    return similar_images, class_names

# function to get similar images considering model weights
def get_similar_images(uploaded_image, mode='clip_trained', weights='logs/clip_model.pth', n=20):
    query_image = uploaded_image  # Directly from the upload

    # load the CLIP model and Faiss index
    if mode == 'clip_trained':
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint)
        
        faiss_index = read_index('clip_trained_ver1_combined_loss.index')

        with open('clip_trained_ver1_combined_loss.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        
        similar_images, class_names = find_similar_images_clip(model, faiss_index, image_paths, query_image, top_k=n)
        return similar_images, class_names

# function to draw bounding boxes on images
def draw_bboxes_on_image(image_path, class_for_test, selected_class_num):
    results = yolo(image_path, conf=0.05)
    filtered_results = []
    
    # filter by class
    for result in results:
        for det in result.boxes:
            if int(det.cls) == selected_class_num:
                filtered_results.append(det)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # draw bounding boxes
    for det in filtered_results:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        class_name = yolo.names[int(det.cls)]
        confidence = det.conf
    
        # draw rectangle and text
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{class_name} {float(confidence):.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img

# function to create a zip archive
def create_zip(images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_path in enumerate(images):
            img = Image.open(img_path)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(f'image_{i+1}.png', img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# streamlit interface
st.set_page_config(page_title='ФотОриентир', page_icon=":mag:", layout='wide')
st.title('Поиск смысловых копий изображений')

uploaded_file = st.file_uploader('Загрузите изображение', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    similar_images, class_names = get_similar_images(original_image, n=20)

    draw_bboxes = st.checkbox("Показывать bounding boxes")

    left_column, right_column = st.columns([1, 2], gap='large')

    with left_column:
        st.subheader('Загруженное изображение')
        st.image(original_image, use_container_width=True, caption='Оригинал')

    with right_column:
        st.subheader('Смысловые копии')

        with st.container():
            columns = st.columns(4)
            for i, img in enumerate(similar_images):
                with columns[i % 4]:
                    if draw_bboxes:
                        img_with_bboxes = draw_bboxes_on_image(img, class_names[i], selected_class_num=class_mapping_dict[class_names[i]])
                        st.image(img_with_bboxes, use_container_width=True, caption=f'Class: {class_names[i]}')
                    else:
                        st.image(img, use_container_width=True, caption=f'Class: {class_names[i]}')

    zip_file = create_zip(similar_images)
    st.download_button(
        label='Скачать архив смысловых копий',
        data=zip_file,
        file_name='similar_images.zip',
        mime='application/zip'
    )
else:
    st.info('Пожалуйста, загрузите изображение.')