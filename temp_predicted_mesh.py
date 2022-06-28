# image tensors as input -> batch_size x height x width x channels
# image tensors shape -> batch_size x 3 x 256 x 256
def generate_mesh(image_tensors):
    predicted_tensor_meshes = list()
    for image_tensor in image_tensors:
        # convert the tensor to np array and generate the mesh landmarks 
        # print(f'Inside generate mesh, shape -> {image_tensor.shape}')

        # permute the tensor 3 x 256 x 256 -> 256 x 256 x 3
        image_tensor = image_tensor.permute(1, 2, 0)

        # convert to np and generate mesh landmarks
        image_array = image_tensor.detach().cpu().numpy()

        # normalize the range (0 -> 1) to (0 -> 255)
        image_array = (image_array * 255).astype(np.uint8)

        image_mesh = mesh_from_image(image_array)

        if image_mesh is None:
            image_mesh = np.empty(image_array.shape).astype(np.uint8)
        else:
            image_mesh = mesh_from_image(image_array).astype(np.uint8)

        # convert the image mesh to tensor 
        predicted_mesh_tensor = transform(image_mesh)

        # print(f'Shape before permutation : {predicted_mesh_tensor.shape}') # 3 x 256 x 256

        predicted_tensor_meshes.append(predicted_mesh_tensor)

    # converted the list of tensors to a global tensor and send 
    concatenated = predicted_tensor_meshes[0].unsqueeze(0)
    for i in range(1, len(predicted_tensor_meshes)):
        concatenated = torch.cat([concatenated, predicted_tensor_meshes[i].unsqueeze(0)], dim=0)

    # print(f'Concatenated shape at inference : {concatenated.shape}')

    return concatenated # dimension -> batch_size x channels x height x width