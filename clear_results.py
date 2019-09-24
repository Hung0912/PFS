import os, shutil

folder = 'results/'
images = 'results/images/'
cluster_centers= 'results/cluster_centers/'
membership_matrixs = 'results/membership_matrixs/'


def clear_files_in_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path):
            #     print(file_path)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    clear_files_in_folder(images)
    clear_files_in_folder(cluster_centers)
    clear_files_in_folder(membership_matrixs)
