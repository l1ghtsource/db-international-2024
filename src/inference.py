import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
import pickle
from faiss import read_index
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from ultralytics import YOLO


def find_similar_images_clip(model, faiss_index, image_paths, query_image, top_k=50, use_classes=True):
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # prepare the image for the CLIP model
    inputs = processor(images=query_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # ensure the model is on the correct device
    model.to(device)

    # extract features
    with torch.no_grad():
        query_embedding = model.get_image_features(**inputs)
        query_embedding = query_embedding.cpu().numpy()[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
    
    # search in the Faiss index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # return the similar images with their paths
    similar_images = [image_paths[i] for i in indices[0]]
    
    # extract class names (parent folder of each image)
    class_names = [image_path.split('/')[-2] for image_path in similar_images]
    
    if use_classes:
        return similar_images, class_names, distances
    else:
        return similar_images, None, distances


def get_similar_images(
    uploaded_image, 
    mode='clip_trained', 
    weights='logs/clip_w_triplet_v2.pth', 
    index='faiss/clip_trained_ver2_triplet_loss', 
    n=10, 
    use_classes=True
):
    query_image = uploaded_image  # directly from the upload

    # load the CLIP model and Faiss index
    if mode == 'clip_trained':
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint)
        
        faiss_index = read_index(f'{index}.index')

        with open(f'{index}.pkl', 'rb') as f:
            image_paths = pickle.load(f)
        
        # first, find top 5n similar images
        similar_images, class_names, distances = find_similar_images_clip(
            model, 
            faiss_index, 
            image_paths, 
            query_image, 
            top_k=5 * n, 
            use_classes=use_classes
        )
        
        if use_classes:
            # determine the dominant class in the top 50
            class_counts = {cls: class_names.count(cls) for cls in set(class_names)}
            dominant_class = max(class_counts, key=class_counts.get)
            
            # now collect only the images from the dominant class
            dominant_images = []
            dominant_classes = []
            for img, cls in zip(similar_images, class_names):
                if cls == dominant_class:
                    dominant_images.append(img)
                    dominant_classes.append(cls)
                if len(dominant_images) == n:
                    break

            # if we don't have enough images from the dominant class, fill up with other images from the same class
            if len(dominant_images) < n:
                # find remaining images from the dominant class
                for img, cls in zip(similar_images[len(dominant_images):], class_names[len(dominant_images):]):
                    if cls == dominant_class:
                        dominant_images.append(img)
                        dominant_classes.append(cls)
                    if len(dominant_images) == n:
                        break
            
            # now we have the final dominant class images
            return dominant_images, dominant_classes
        else:
            # return top n images without considering classes
            return similar_images[:n], None


def inference(
    image_folder, 
    output_csv='submission.csv', 
    mode='clip_trained', 
    weights='logs/clip_w_triplet_v2.pth', 
    index='faiss/clip_trained_ver2_triplet_loss', 
    n=10, 
    use_classes=True
):
    # list all image files in the given folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    # prepare the list to store results
    results = []

    # wrap the loop with tqdm for progress tracking
    for image_file in tqdm(image_files, desc="Processing images", ncols=100):
        image_path = os.path.join(image_folder, image_file)

        # open the image
        uploaded_image = Image.open(image_path)

        # find similar images for the current image
        similar_images, class_names = get_similar_images(
            uploaded_image, 
            mode=mode, 
            weights=weights, 
            index=index, 
            n=n, 
            use_classes=use_classes
        )

        # extract the image name without extension
        image_name = image_file

        # get the list of recommended image names without extensions
        recommended_images = [os.path.basename(img) for img in similar_images]

        # join the recommended images with commas
        recs = ','.join(recommended_images)

        # append the result
        results.append({'image': image_name, 'recs': f'{recs}'})

    # create a DataFrame and save it to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"Submission CSV saved to {output_csv}")
    
def find_similar_images_yolo(model, faiss_index, image_paths, query_image, top_k=50, use_classes=True):
    # prepare the image for YOLO model
    image = Image.open(query_image).convert('RGB')

    # get the embedding from the YOLO model
    embedding = model.embed(query_image)[0].cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding)  # normalize the feature vector
    embedding = embedding.reshape(1, -1).astype("float32")
    
    # search in the Faiss index
    distances, indices = faiss_index.search(embedding, top_k)
    
    # return the similar images with their paths
    similar_images = [image_paths[i] for i in indices[0]]
    
    # extract class names (parent folder of each image)
    class_names = [image_path.split('/')[-2] for image_path in similar_images]
    
    if use_classes:
        return similar_images, class_names, distances
    else:
        return similar_images, None, distances


def get_similar_images_yolo(
    uploaded_image, 
    model, 
    index='faiss/yolo_index', 
    n=10, 
    use_classes=True
):
    query_image = uploaded_image  # directly from the upload

    # load the Faiss index for YOLO
    faiss_index = read_index(f'{index}.index')

    with open(f'{index}.pkl', 'rb') as f:
        image_paths = pickle.load(f)

    # find top 5n similar images
    similar_images, class_names, distances = find_similar_images_yolo(
        model, 
        faiss_index, 
        image_paths, 
        query_image, 
        top_k=5 * n, 
        use_classes=use_classes
    )
    
    if use_classes:
        # determine the dominant class in the top 50
        class_counts = {cls: class_names.count(cls) for cls in set(class_names)}
        dominant_class = max(class_counts, key=class_counts.get)
        
        # now collect only the images from the dominant class
        dominant_images = []
        dominant_classes = []
        for img, cls in zip(similar_images, class_names):
            if cls == dominant_class:
                dominant_images.append(img)
                dominant_classes.append(cls)
            if len(dominant_images) == n:
                break

        # if we don't have enough images from the dominant class, fill up with other images from the same class
        if len(dominant_images) < n:
            for img, cls in zip(similar_images[len(dominant_images):], class_names[len(dominant_images):]):
                if cls == dominant_class:
                    dominant_images.append(img)
                    dominant_classes.append(cls)
                if len(dominant_images) == n:
                    break
        
        # now we have the final dominant class images
        return dominant_images, dominant_classes
    else:
        # return top n images without considering classes
        return similar_images[:n], None


def inference_yolo(
    image_folder, 
    model, 
    output_csv='submission_yolo.csv', 
    index='faiss/yolo_index', 
    n=10, 
    use_classes=True
):
    # list of all image files in the given folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    # prepare the list to store results
    results = []

    # wWrap the loop with tqdm for progress tracking
    for image_file in tqdm(image_files, desc="Processing images", ncols=100):
        image_path = os.path.join(image_folder, image_file)

        # find similar images for the current image
        similar_images, class_names = get_similar_images_yolo(
            uploaded_image=image_path, 
            model=model, 
            index=index, 
            n=n, 
            use_classes=use_classes
        )

        # extract the image name without extension
        image_name = image_file

        # get the list of recommended image names without extensions
        recommended_images = [os.path.basename(img) for img in similar_images]
        
        # join the recommended images with commas
        recs = ','.join(recommended_images)

        # append the result
        results.append({'image': image_name, 'recs': f'{recs}'})

    # create a DataFrame and save it to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    print(f"Submission CSV saved to {output_csv}")
    
    
def find_similar_images_combined(
    clip_weights, yolo_model, faiss_index, image_paths, query_image, top_k=50, use_classes=True
):
    # initialize CLIP and YOLO processors
    clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
    checkpoint = torch.load(clip_weights)
    clip_model.load_state_dict(checkpoint)
    
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # move models to the appropriate device
    clip_model.to(device)
    
    # process the image for CLIP
    clip_inputs = processor(images=query_image, return_tensors="pt")
    clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
    with torch.no_grad():
        clip_embedding = clip_model.get_image_features(**clip_inputs).cpu().numpy()[0]
    
    # get YOLO embedding
    yolo_embedding = yolo_model.embed(query_image)[0].cpu().numpy()
    
    # concatenate and normalize
    query_embedding = np.concatenate((clip_embedding, yolo_embedding))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    
    # search in the Faiss index
    distances, indices = faiss_index.search(query_embedding, top_k)
    
    # return paths to similar images and their classes
    similar_images = [image_paths[i] for i in indices[0]]
    class_names = [image_path.split('/')[-2] for image_path in similar_images]
    
    if use_classes:
        return similar_images, class_names, distances
    else:
        return similar_images, None, distances


def get_similar_images_combined(
    uploaded_image, clip_weights, yolo_model, index='faiss/combined_index', n=10, use_classes=True
):
    # load the Faiss index and image paths
    faiss_index = read_index(f'{index}.index')
    with open(f'{index}.pkl', 'rb') as f:
        image_paths = pickle.load(f)

    # find the top-5*n similar images to analyze the dominant class
    similar_images, class_names, distances = find_similar_images_combined(
        clip_weights, yolo_model, faiss_index, image_paths, uploaded_image, top_k=5 * n, use_classes=use_classes
    )
    
    if use_classes:
        # determine the dominant class in the results
        class_counts = {cls: class_names.count(cls) for cls in set(class_names)}
        dominant_class = max(class_counts, key=class_counts.get)
        
        # build a list of images from the dominant class
        dominant_images, dominant_classes = [], []
        for img, cls in zip(similar_images, class_names):
            if cls == dominant_class:
                dominant_images.append(img)
                dominant_classes.append(cls)
            if len(dominant_images) == n:
                break
        
        # if there are not enough images, add the remaining ones from the dominant class
        if len(dominant_images) < n:
            for img, cls in zip(similar_images[len(dominant_images):], class_names[len(dominant_images):]):
                if cls == dominant_class:
                    dominant_images.append(img)
                    dominant_classes.append(cls)
                if len(dominant_images) == n:
                    break
        
        return dominant_images, dominant_classes
    else:
        # if classes are not considered, return the top-n images
        return similar_images[:n], None


def inference_combined(
    image_folder, clip_weights, yolo_model, output_csv='submission_combined.csv', index='faiss/combined_index', 
    n=10, use_classes=True
):
    # load all images from the specified folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    results = []
    for image_file in tqdm(image_files, desc="Processing images", ncols=100):
        image_path = os.path.join(image_folder, image_file)
        uploaded_image = Image.open(image_path)

        # find similar images
        similar_images, class_names = get_similar_images_combined(
            uploaded_image=uploaded_image, clip_weights=clip_weights, yolo_model=yolo_model, 
            index=index, n=n, use_classes=use_classes
        )

        # format the results
        image_name = image_file
        recommended_images = [os.path.basename(img) for img in similar_images]
        recs = ','.join(recommended_images)
        results.append({'image': image_name, 'recs': f'{recs}'})

    # save the results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Submission CSV saved to {output_csv}")


# inference_combined(
#     image_folder='images', 
#     clip_weights='logs/clip_model.pth', 
#     yolo_model=YOLO('yolov8x-oiv7.pt'), 
#     output_csv='submission_concat.csv', 
#     index='faiss/combined_index', 
#     n=10, 
#     use_classes=True
# )
    
# inference(
#     image_folder='images', 
#     output_csv='submission_combined_ver1_use_classes.csv', 
#     mode='clip_trained', 
#     weights='logs/clip_model.pth', 
#     index='faiss/clip_trained_ver1_combined_loss', 
#     n=10,
#     use_classes=True
# )
  
# inference(
#     image_folder='images', 
#     output_csv='submission_triplet_ver2_use_classes.csv', 
#     mode='clip_trained', 
#     weights='logs/clip_w_triplet_v2.pth', 
#     index='faiss/clip_trained_ver2_triplet_loss', 
#     n=10,
#     use_classes=True
# )

# inference_yolo(
#     image_folder='images', 
#     model=YOLO('yolov8x-oiv7.pt'),
#     output_csv='submission_yolo_use_classes.csv', 
#     index='faiss/yolo_index', 
#     n=10, 
#     use_classes=True
# )

# inference(
#     image_folder='images', 
#     output_csv='submission_combined_ver1_no_classes.csv', 
#     mode='clip_trained', 
#     weights='logs/clip_model.pth', 
#     index='faiss/clip_trained_ver1_combined_loss', 
#     n=10,
#     use_classes=False
# )

# inference(
#     image_folder='images', 
#     output_csv='submission_triplet_ver2_no_classes.csv', 
#     mode='clip_trained', 
#     weights='logs/clip_w_triplet_v2.pth', 
#     index='faiss/clip_trained_ver2_triplet_loss', 
#     n=10,
#     use_classes=False
# )

# inference_yolo(
#     image_folder='images', 
#     model=YOLO('yolov8x-oiv7.pt'),
#     output_csv='submission_yolo_no_classes.csv', 
#     index='faiss/yolo_index', 
#     n=10, 
#     use_classes=False
# )
