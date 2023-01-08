#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/c_api.h>

int main(int argc, char* argv[]) {

  // Load the pre-trained model
  TF_Buffer* model_buffer = read_file("mobilenet_v2_1.0_224_frozen.pb");
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, model_buffer, options, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error importing model: %s\n", TF_Message(status));
    exit(1);
  }
  TF_DeleteImportGraphDefOptions(options);
  TF_DeleteBuffer(model_buffer);

  // Pre-process an image of a leukocyte
  float* preprocess_image(const char* image_path, int* width, int* height, int* channels) {
    // Load the image
    int x, y, n;
    unsigned char* data = stbi_load(image_path, &x, &y, &n, 3);
    if (!data) {
      fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
      exit(1);
    }

    // Pre-process the image by subtracting the mean pixel value and scaling
    float* image = malloc(x * y * 3 * sizeof(float));
    for (int i = 0; i < x * y * 3; i++) {
      image[i] = (data[i] - 128) / 128.0;
    }
    free(data);

    *width = x;
    *height = y;
    *channels = 3;
    return image;
  }

  // Load an image of a leukocyte
  int width, height, channels;
  float* image = preprocess_image("leukocyte.jpg", &width, &height, &channels);

  // Create input tensor
  TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, &width, 4, 3);
  memcpy(TF_TensorData(input_tensor), image, width * height * channels * sizeof(float));

  // Run the model
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sess_opts, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "Error creating session: %s\n", TF_Message(status));
    exit(1);
  }
  TF_Operation* input_op = TF_GraphOperationByName(graph, "input_1");
  // TF_Output
  
}
