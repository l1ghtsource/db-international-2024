import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import time
import numpy as np
import faiss
from faiss import write_index
import matplotlib.pyplot as plt
import pickle
from ultralytics import YOLO


embeddings = []  # list to store image feature embeddings
image_paths = []  # list to store image file paths


def load_and_vectorize_images_clip(model, dataset_path):
    start_time = time.time()
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')  # initialize the CLIP processor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # choose device (GPU or CPU)
    model = model.to(device)  # move model to selected device
    model.eval()  # set the model to evaluation mode

    for class_ in tqdm(os.listdir(dataset_path)):  # iterate through classes in the dataset
        class_path = os.path.join(dataset_path, class_)
        
        # skip non-directory files (e.g., .DS_Store)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):  # iterate through files in each class
            image_path = os.path.join(class_path, file)
            try:
                image = Image.open(image_path).convert('RGB')  # open and convert the image to RGB
                inputs = processor(images=image, return_tensors='pt')  # process image for CLIP model
                
                inputs = {k: v.to(device) for k, v in inputs.items()}  # move inputs to the device
                
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)  # extract image features

                embedding = image_features.cpu().numpy()[0]  # convert features to numpy array
                embedding = embedding / np.linalg.norm(embedding)  # normalize the feature vector
                
                embeddings.append(embedding)  # add the embedding to the list
                image_paths.append(image_path)  # add the image path to the list
                
            except Exception as e:
                print(f'Error processing {image_path}: {e}')  # handle any errors

    end_time = time.time()
    print(f'Image loading and vectorization completed in {end_time - start_time:.2f} seconds')  # print time taken
    
    return embeddings, image_paths  # return the embeddings and image paths


def create_faiss_index_clip(embeddings, dimension):
    start_time = time.time()
    
    embeddings_array = np.array(embeddings).astype('float32')  # convert the embeddings list to a numpy array
    index = faiss.IndexFlatL2(dimension)  # create a FAISS index using L2 distance
    index.add(embeddings_array)  # add the embeddings to the index
    
    end_time = time.time()
    print(f'FAISS index creation completed in {end_time - start_time:.2f} seconds')  # print time taken
    
    return index  # return the FAISS index
    

def save_index_clip(trained=True, weights='logs/clip_w_triplet_v2.pth', index='./faiss/clip_trained_ver2_triplet_loss'):
    pt_clip_v2 = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')  # load the CLIP model
    if trained:
        checkpoint = torch.load(weights)  # load the trained weights
        pt_clip_v2.load_state_dict(checkpoint)  # load the model state

    dataset_path = '../data/train_data_rkn/dataset'  # path to the dataset
    clip_embs_v2, clip_images_v2 = load_and_vectorize_images_clip(pt_clip_v2, dataset_path)  # vectorize images
    dimension = 512  # feature dimension
    index_clip_v2 = create_faiss_index_clip(clip_embs_v2, dimension)  # create FAISS index
    
    # save the index and image paths
    if trained:
        write_index(index_clip_v2, f'{index}.index')
        with open(f'{index}.pkl', 'wb') as f:
            pickle.dump(clip_images_v2, f)
        
    else:
        write_index(index_clip_v2, './faiss/clip_default.index')
        with open('./faiss/clip_default.pkl', 'wb') as f:
            pickle.dump(clip_images_v2, f)
            
            
def load_and_vectorize_images_yolo(model, dataset_path):
    start_time = time.time()

    for class_ in tqdm(os.listdir(dataset_path)):  # iterate through classes in the dataset
        class_path = os.path.join(dataset_path, class_)
        
        # skip non-directory files (e.g., .DS_Store)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):  # iterate through files in each class
            image_path = os.path.join(class_path, file)
            try:
                embedding = model.embed(image_path)[0]  # get embedding from YOLO model
                
                # move the tensor to CPU and convert to numpy
                embedding = embedding.cpu().numpy()  # ensure tensor is moved to CPU before converting to numpy
                embedding = embedding / np.linalg.norm(embedding)  # normalize the feature vector
                
                embeddings.append(embedding)  # add the embedding to the list
                image_paths.append(image_path)  # add the image path to the list
            except Exception as e:
                print(f'Error processing {image_path}: {e}')  # handle any errors

    end_time = time.time()
    print(f'Image loading and vectorization completed in {end_time - start_time:.2f} seconds')  # print time taken
    
    return embeddings, image_paths  # return the embeddings and image paths


def create_faiss_index_yolo(embeddings, dimension):
    start_time = time.time()
    
    # convert the embeddings list to a numpy array (ensure the embeddings are in float32 format)
    embeddings_array = np.array([embedding for embedding in embeddings]).astype('float32')
    
    # create a FAISS index using L2 distance
    index = faiss.IndexFlatL2(dimension)  
    index.add(embeddings_array)  # Add the embeddings to the index
    
    end_time = time.time()
    print(f'FAISS index creation completed in {end_time - start_time:.2f} seconds')  # print time taken
    
    return index  # return the FAISS index
    
    
def save_index_yolo(index='./faiss/yolo_index'):
    model = YOLO('yolov8x-oiv7.pt')
    dataset_path = '../data/train_data_rkn/dataset'

    # vectorize images using YOLO
    embs_yolo, image_paths_yolo = load_and_vectorize_images_yolo(model, dataset_path)
    # determine the dimensionality of the embeddings
    dimension = embs_yolo[0].shape[0]
    # create FAISS index
    faiss_index_yolo = create_faiss_index_yolo(embs_yolo, dimension)

    # save the FAISS index and image paths
    write_index(faiss_index_yolo, f'{index}.index')  # save the FAISS index
    with open(f'{index}.pkl', 'wb') as f:
        pickle.dump(image_paths_yolo, f)  # save the image paths
     
        
def load_and_vectorize_images_combined(clip_model, yolo_model, dataset_path):
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    embeddings_combined = []
    image_paths_combined = []
    
    for class_ in tqdm(os.listdir(dataset_path)):
        class_path = os.path.join(dataset_path, class_)
        if not os.path.isdir(class_path):
            continue
        
        for file in os.listdir(class_path):
            image_path = os.path.join(class_path, file)
            try:
                # process image with CLIP
                image = Image.open(image_path).convert('RGB')
                clip_inputs = clip_processor(images=image, return_tensors='pt')
                clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
                
                with torch.no_grad():
                    clip_emb = clip_model.get_image_features(**clip_inputs).cpu().numpy()[0]
                
                # process image with YOLO
                yolo_emb = yolo_model.embed(image_path)[0].cpu().numpy()
                
                # concatenate embeddings and normalize
                combined_emb = np.concatenate((clip_emb, yolo_emb))
                combined_emb = combined_emb / np.linalg.norm(combined_emb)
                
                # append to lists
                embeddings_combined.append(combined_emb)
                image_paths_combined.append(image_path)
                
            except Exception as e:
                print(f'Error processing {image_path}: {e}')
    
    return embeddings_combined, image_paths_combined


def create_faiss_index_combined(embeddings_combined, dimension):
    embeddings_array = np.array(embeddings_combined).astype('float32')
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    return index


def save_index_combined(clip_weights='logs/clip_w_triplet_v2.pth', yolo_weights='yolov8x-oiv7.pt', index_path='./faiss/combined_index'):
    # load models
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    checkpoint = torch.load(clip_weights)
    clip_model.load_state_dict(checkpoint)
    yolo_model = YOLO(yolo_weights)
    
    # dataset path
    dataset_path = '../data/train_data_rkn/dataset'
    
    # vectorize images
    combined_embs, combined_image_paths = load_and_vectorize_images_combined(clip_model, yolo_model, dataset_path)
    
    # create FAISS index
    dimension = combined_embs[0].shape[0]
    faiss_index_combined = create_faiss_index_combined(combined_embs, dimension)
    
    # save FAISS index and image paths
    write_index(faiss_index_combined, f'{index_path}.index')
    with open(f'{index_path}.pkl', 'wb') as f:
        pickle.dump(combined_image_paths, f)


# it was already saved
# save_index_clip(trained=True, weights='logs/clip_w_triplet_v2.pth', index='./faiss/clip_trained_ver2_triplet_loss')  
# save_index_clip(trained=True, weights='logs/clip_model.pth', index='./faiss/clip_trained_ver1_combined_loss')  
# save_index_yolo(index='./faiss/yolo_index')
# save_index_clip(trained=True, weights='logs/clip_triplet_tuned.pth', index='./faiss/clip_trained_ver3_triplet_loss')  
# save_index_combined(
#     clip_weights='logs/clip_model.pth',
#     yolo_weights='yolov8x-oiv7.pt', 
#     index_path='./faiss/combined_index'
# )
