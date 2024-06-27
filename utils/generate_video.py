import os
import cv2

def images_to_video(image_folder, output_video_path, fps=30):
    image_files = [file for file in os.listdir(image_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    if not image_files:
        print("No image files found in the folder.")
        return

    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved to: {output_video_path}")


if __name__ == '__main__':
    image_folder_path = '/data/yqge/miao/DynamicNeRF/data/blackswan/images'
    output_video_path = '/data/yqge/miao/DynamicNeRF/data/blackswan/blackswan.mp4'
    fps = 30

    images_to_video(image_folder_path, output_video_path, fps)