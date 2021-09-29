#include <iostream>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <GL/gl.h>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <utility>

#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#define IM_F32_TO_INT8_SAT(_VAL)                                               \
  ((int)(ImSaturate(_VAL) * 255.0f + 0.5f)) // Saturated, always output 0..255
static inline float
ImSaturate(float f)
{
  return (f < 0.0f) ? 0.0f : (f > 1.0f) ? 1.0f : f;
}

using namespace cv;

void
show_image(Mat const& image);

GLFWwindow*
init(uint width, uint height);

static void
glfw_error_callback(int error, const char* description);

void
renderUI(bool& is_show, Mat image)
{
  // state
  static int threshValue = 15;
  static float color[3] = { 1.0f, 1.0f, 0.9f };
  static int maxThresh = 255;
  // Lab select face: 145, 21, 21, thresh: 42

  static const auto format = std::map<std::string, std::tuple<int, int>>{
    { "rgb", { COLOR_BGR2RGB, COLOR_RGB2BGR } },
    { "hls", { COLOR_BGR2HLS, COLOR_HLS2BGR } },
    { "lab", { COLOR_BGR2Lab, COLOR_Lab2BGR } },
    { "gray", { COLOR_BGR2GRAY, COLOR_GRAY2BGR}}, 
  };

  static std::string value = "hls";
  auto [encode, decode] = format.at(value);

  Mat imageInEncoding, grayImage, resultGrayImage;
  Mat mask, resultImage;
  Mat3b threshColorInEncoding;

  // move from BGR to threshColorInEncoding
  cvtColor(image, imageInEncoding, encode);
  cvtColor(image, grayImage, COLOR_BGR2GRAY);

  // BGR color is swaps R and B
  Mat3b threshColor(cv::Vec3b(IM_F32_TO_INT8_SAT(color[2]),
                              IM_F32_TO_INT8_SAT(color[1]),
                              IM_F32_TO_INT8_SAT(color[0])));

  cvtColor(threshColor, threshColorInEncoding, encode);

  Vec3b pixelInEncoding(threshColorInEncoding.at<Vec3b>(0, 0));

  // make mask
  auto minMaskColor = cv::Scalar(pixelInEncoding.val[0] - threshValue,
                                 pixelInEncoding.val[1] - threshValue,
                                 pixelInEncoding.val[2] - threshValue);
  auto maxMaskColor = cv::Scalar(pixelInEncoding.val[0] + threshValue,
                                 pixelInEncoding.val[1] + threshValue,
                                 pixelInEncoding.val[2] + threshValue);
  cv::inRange(imageInEncoding, minMaskColor, maxMaskColor, mask);

  std::cout << "color: x" << IM_F32_TO_INT8_SAT(color[0]) << " y"
            << IM_F32_TO_INT8_SAT(color[1]) << " z"
            << IM_F32_TO_INT8_SAT(color[2]) << std::endl
            << ". threshColor: " << threshColor << std::endl;

  // apply mask to source image
  cv::bitwise_and(imageInEncoding, imageInEncoding, resultImage, mask);
  cv::threshold(grayImage, resultGrayImage, threshValue, maxThresh, THRESH_BINARY); 

  std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(resultGrayImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	// draw contours on the original image
	Mat image_copy = image.clone();
	drawContours(image_copy, contours, -1, Scalar(0, 255, 0), 2);

  // fix for imgui
  cvtColor(image, image, COLOR_BGR2RGBA);
  cvtColor(image_copy, image_copy, COLOR_BGR2RGBA);
  cvtColor(resultImage, resultImage, decode);
  cvtColor(resultImage, resultImage, COLOR_BGR2RGBA);
  cvtColor(resultGrayImage, resultGrayImage, COLOR_GRAY2RGBA);



  ImGui::Begin("image", &is_show, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::SetWindowFontScale(2);
  ImGui::ColorEdit3("tresh color", color);
  ImGui::DragInt("thresh", &threshValue, 1, 0, 255);
  ImGui::DragInt("maxThresh", &maxThresh, 1, 0, 255);
  if (ImGui::Button("exit")) {
    is_show = false;
  }
  show_image(image);
  ImGui::SameLine();
  show_image(resultImage);
  ImGui::SameLine();
  show_image(resultGrayImage);
  ImGui::SameLine();
  show_image(image_copy);

  ImGui::End();
}

int
main(int argc, char** argv)
{
  if (argv[1] == NULL)
    return 1;

  Mat image = imread(argv[1]);

  auto window = init(image.cols, image.rows);

  bool is_show = true;

  while (is_show) {
    glfwPollEvents();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    renderUI(is_show, image);

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }

  ImGui_ImplGlfw_Shutdown();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui::DestroyContext();
  glfwTerminate();
  return 0;
}

void
show_image(Mat const& image)
{
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGBA,
               image.cols,
               image.rows,
               0,
               GL_RGBA,
               GL_UNSIGNED_BYTE,
               image.data);

  ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(texture)),
               ImVec2(image.cols, image.rows));
}

static void
glfw_error_callback(int error, const char* description)
{
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

GLFWwindow*
init(uint width, uint height)
{
  if (!glfwInit()) {
    return nullptr;
  }
  GLFWwindow* window =
    glfwCreateWindow(width, height, "glfw window", nullptr, nullptr);

  if (window == nullptr) {
    return nullptr;
  }

  glfwSetWindowCloseCallback(window, [](GLFWwindow* window) {
    glfwSetWindowShouldClose(window, GL_FALSE);
  });

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  return window;
}
