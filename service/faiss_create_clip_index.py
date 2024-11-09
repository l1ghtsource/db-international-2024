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


def visualize_similar_images_clip(query_image_path, similar_images):
    query_image = Image.open(query_image_path)  # open the query image
    similar_images = [Image.open(img_path) for img_path in similar_images]  # open the similar images
    fig = plt.figure(figsize=(9, 9))  # create a figure for visualization

    # display the query image
    plt.subplot(3, 3, 1)
    plt.imshow(query_image)
    plt.title('Query Image', fontsize=12, pad=10)
    plt.axis('off')

    # display the similar images
    for i, img in enumerate(similar_images[1:9]): 
        plt.subplot(3, 3, i + 2)
        plt.imshow(img)
        plt.title(f'Similar {i + 1}', fontsize=12, pad=10)
        plt.axis('off')

    plt.tight_layout()  # adjust layout
    plt.show()  # display the visualization
    

def save_index(trained=True, weights='logs/clip_model.pth'):
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
        write_index(index_clip_v2, './clip_trained_ver1_combined_loss.index')
        with open('./clip_trained_ver1_combined_loss.pkl', 'wb') as f:
            pickle.dump(clip_images_v2, f)
        
    else:
        write_index(index_clip_v2, './clip_default.index')
        with open('./clip_default.pkl', 'wb') as f:
            pickle.dump(clip_images_v2, f)

save_index()  # call the function to save the index