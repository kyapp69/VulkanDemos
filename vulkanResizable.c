#include <stdio.h>
#include <math.h>
#ifndef M_PI
  #define M_PI 3.14159265358979323846  
#endif /* M_PI */
#define VK_USE_PLATFORM_XCB_KHR
#include "vulkan/vulkan.h"
//#include "vulkan/vk_sdk_platform.h"
#include <xcb/xcb.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
//#include <dlfcn.h>
#include <unistd.h>

#define NUM_SAMPLES 1
    
struct Vertex{
    float posX, posY, posZ, posW; // Position data
    float r, g, b, a;             // Color
};

#define XYZ1(_x_, _y_, _z_) (_x_), (_y_), (_z_), 1.f

static const struct Vertex vertexData[] = {
    {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},

    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},

    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},

    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)},

    {XYZ1(1, 1, 1), XYZ1(1.f, 1.f, 1.f)},
    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(1, 1, -1), XYZ1(1.f, 1.f, 0.f)},
    {XYZ1(-1, 1, 1), XYZ1(0.f, 1.f, 1.f)},
    {XYZ1(-1, 1, -1), XYZ1(0.f, 1.f, 0.f)},

    {XYZ1(1, -1, 1), XYZ1(1.f, 0.f, 1.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(-1, -1, 1), XYZ1(0.f, 0.f, 1.f)},
    {XYZ1(1, -1, -1), XYZ1(1.f, 0.f, 0.f)},
    {XYZ1(-1, -1, -1), XYZ1(0.f, 0.f, 0.f)},
};

//Load binary file into a buffer
char *readBinaryFile(const char *filename, size_t *psize)
{
	long int size;
	size_t retval;
	void *fileContents;

	FILE *fileHandle = fopen(filename, "rb");
	if (!fileHandle) return NULL;

	fseek(fileHandle, 0L, SEEK_END);
	size = ftell(fileHandle);

	fseek(fileHandle, 0L, SEEK_SET);

	fileContents = malloc(size);
	retval = fread(fileContents, size, 1, fileHandle);
	assert(retval == 1);

	*psize = size;

	return (char*)fileContents;
}

/*
 * Simulates gluPerspectiveMatrix
 */
void perspective_matrix(double fovy, double aspect, double znear, double zfar, float *P) {
    int i;
    double f;

    f = 1.0/tan(fovy * 0.5);

    for (i = 0; i < 16; i++) {
        P[i] = 0.0;
    }

    P[0] = f / aspect;
    P[5] = f;
    P[10] = (znear + zfar) / (znear - zfar);
    P[11] = -1.0;
    P[14] = (2.0 * znear * zfar) / (znear - zfar);
    P[15] = 0.0;
}
/*
 * Multiplies A by B and writes out to C. All matrices are 4x4 and column
 * major. In-place multiplication is supported.
 */
void multiply_matrix(float *A, float *B, float *C) {
	int i, j, k;
    float aTmp[16];

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            aTmp[j * 4 + i] = 0.0;

            for (k = 0; k < 4; k++) {
                aTmp[j * 4 + i] += A[k * 4 + i] * B[j * 4 + k];
            }
        }
    }

    memcpy(C, aTmp, sizeof(aTmp));
}

void identity_matrix(float *matrix) {
    float aTmp[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    memcpy(matrix, aTmp, sizeof(aTmp));
}

void translate_matrix(double x, double y, double z, float *matrix) {
    identity_matrix(matrix);
    matrix[12]=x;
    matrix[13]=y;
    matrix[14]=z;
}

/*
 * Simulates desktop's glRotatef. The matrix is returned in column-major
 * order.
 */
void rotate_matrix(double angle, double x, double y, double z, float *R) {
    double radians, c, s, c1, u[3], length;
    int i, j;

    radians = (angle * M_PI) / 180.0;

    c = cos(radians);
    s = sin(radians);

    c1 = 1.0 - cos(radians);

    length = sqrt(x * x + y * y + z * z);

    u[0] = x / length;
    u[1] = y / length;
    u[2] = z / length;

    for (i = 0; i < 16; i++) {
        R[i] = 0.0;
    }

    R[15] = 1.0;

    for (i = 0; i < 3; i++) {
        R[i * 4 + (i + 1) % 3] = u[(i + 2) % 3] * s;
        R[i * 4 + (i + 2) % 3] = -u[(i + 1) % 3] * s;
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            R[i * 4 + j] += c1 * u[i] * u[j] + (i == j ? c : 0.0);
        }
    }
}

void print_matrix(float *matrix) {
    int i;
    //Print matrix stored in column-major format using stadard maths layout:
    for (i = 0; i < 4; i++) {
        printf("%f, %f, %f, %f\n", matrix[0*4+i], matrix[1*4+i], matrix[2*4+i], matrix[3*4+i]);
    }
}

int width = 640;
int height = 480;

int initSwapchain(VkPhysicalDevice *physicalDevices, VkDevice device, VkSurfaceKHR surface, VkSwapchainKHR *pSwapchain, VkFormat *pFormat)
{
  VkSwapchainKHR oldSwapchain = *pSwapchain;
  VkSurfaceCapabilitiesKHR surfCapabilities;  
  VkResult res;
  
  //Setup the swapchain
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevices[0], surface, &formatCount, NULL);
  VkSurfaceFormatKHR formats[formatCount];
  vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevices[0], surface, &formatCount, formats);
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevices[0], surface, &presentModeCount, NULL);
  VkPresentModeKHR presentModes[presentModeCount];
  vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevices[0], surface, &presentModeCount, presentModes);
  res = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevices[0], surface,
                                                    &surfCapabilities);
  
  VkExtent2D swapChainExtent;
  // width and height are either both -1, or both not -1.
  if (surfCapabilities.currentExtent.width == (uint32_t)-1) {
    // If the surface size is undefined, the size is set to
    // the size of the images requested.    
    swapChainExtent.width = width;
    swapChainExtent.height = height;
    printf("Swapchain size is (-1, -1)\n");
  } else {
    // If the surface size is defined, the swap chain size must match
    swapChainExtent = surfCapabilities.currentExtent;
    printf("Swapchain size is (%d, %d)\n", swapChainExtent.width, swapChainExtent.height);
  }
    
  VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
  
  // If the format list includes just one entry of VK_FORMAT_UNDEFINED,
  // the surface has no preferred format.  Otherwise, at least one
  // supported format will be returned.
  if (formatCount == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
      *pFormat = VK_FORMAT_B8G8R8A8_UNORM;
  } else {
      assert(formatCount >= 1);
      *pFormat = formats[0].format;
  }
  printf("Using format %d\n", *pFormat);
    
  uint32_t desiredNumberOfSwapChainImages = surfCapabilities.minImageCount + 1;
  if ((surfCapabilities.maxImageCount > 0) &&
      (desiredNumberOfSwapChainImages > surfCapabilities.maxImageCount)) {
      // Application must settle for fewer images than desired:
      desiredNumberOfSwapChainImages = surfCapabilities.maxImageCount;
  }  
  printf("Asking for %d SwapChainImages\n", desiredNumberOfSwapChainImages);
  
  VkSurfaceTransformFlagBitsKHR preTransform;
  if (surfCapabilities.supportedTransforms &
    VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  } else {
    preTransform = surfCapabilities.currentTransform;
  }  
  printf("Using preTransform %d\n", preTransform);
  
  VkSwapchainCreateInfoKHR swapCreateInfo;
  swapCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapCreateInfo.pNext = NULL;
  swapCreateInfo.surface = surface;
  swapCreateInfo.minImageCount = desiredNumberOfSwapChainImages;
  swapCreateInfo.imageFormat = *pFormat;
  swapCreateInfo.imageExtent=swapChainExtent;
  //swapCreateInfo.imageExtent.width = width; //Should match window size
  //swapCreateInfo.imageExtent.height = height;
  swapCreateInfo.preTransform = preTransform;
  swapCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapCreateInfo.imageArrayLayers = 1;
  swapCreateInfo.presentMode = swapchainPresentMode;
  swapCreateInfo.oldSwapchain = oldSwapchain;
  swapCreateInfo.clipped = VK_TRUE;
  swapCreateInfo.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  swapCreateInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  swapCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  swapCreateInfo.queueFamilyIndexCount = 0;
  swapCreateInfo.pQueueFamilyIndices = NULL;
  
  vkCreateSwapchainKHR(device, &swapCreateInfo, NULL, pSwapchain);
  if (res != VK_SUCCESS) {
    printf ("vkCreateSwapchainKHR returned error.\n");
    return -1;
  }   
  if (oldSwapchain!=VK_NULL_HANDLE)
    vkDestroySwapchainKHR(device, oldSwapchain, 0);
}

int initColourBuffers(VkPhysicalDevice *physicalDevices, VkDevice device, VkSwapchainKHR swapchain, VkFormat format, VkCommandBuffer setupBuffer, VkImage **pSwapChainImages, VkImageView **pSwapChainViews, int *pSwapchainImageCount)
{  
  VkResult res;
  
  res = vkGetSwapchainImagesKHR(device,swapchain, pSwapchainImageCount, NULL);
  if (res != VK_SUCCESS) {
    printf ("vkCreateSwapchainKHR returned error.\n");
    return -1;
  } 
  int swapchainImageCount=*pSwapchainImageCount;  
  VkImage *swapChainImages=malloc(sizeof(VkImage)*swapchainImageCount);
  *pSwapChainImages=swapChainImages;
  res = vkGetSwapchainImagesKHR(device, swapchain, pSwapchainImageCount, swapChainImages);  
  if (res != VK_SUCCESS) {
    printf ("vkCreateSwapchainKHR returned error.\n");
    return -1;
  }  
  
  printf ("swapchainImageCount %d.\n",swapchainImageCount);
  VkImageView *swapChainViews = malloc(sizeof(VkImageView)*swapchainImageCount);
  *pSwapChainViews=swapChainViews;
  for (uint32_t i = 0; i < swapchainImageCount; i++) {
    printf ("Setting up swapChainView for swapchain image %d.\n", swapChainImages[i]);
    VkImageViewCreateInfo color_image_view = {};
    color_image_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    color_image_view.pNext = NULL;
    color_image_view.format = format;
    color_image_view.components.r = VK_COMPONENT_SWIZZLE_R;
    color_image_view.components.g = VK_COMPONENT_SWIZZLE_G;
    color_image_view.components.b = VK_COMPONENT_SWIZZLE_B;
    color_image_view.components.a = VK_COMPONENT_SWIZZLE_A;
    color_image_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    color_image_view.subresourceRange.baseMipLevel = 0;
    color_image_view.subresourceRange.levelCount = 1;
    color_image_view.subresourceRange.baseArrayLayer = 0;
    color_image_view.subresourceRange.layerCount = 1;
    color_image_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    color_image_view.flags = 0;
    color_image_view.image = swapChainImages[i];

    VkImageMemoryBarrier imageMemoryBarrier;
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    imageMemoryBarrier.pNext = NULL;
    imageMemoryBarrier.image = swapChainImages[i];
    imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    imageMemoryBarrier.subresourceRange.levelCount = 1;
    imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    imageMemoryBarrier.subresourceRange.layerCount = 1;
    imageMemoryBarrier.srcQueueFamilyIndex=0;
    imageMemoryBarrier.dstQueueFamilyIndex=0;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = 0;

    // Put barrier on top
    VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // Put barrier inside setup command buffer
    vkCmdPipelineBarrier(setupBuffer, srcStageFlags, destStageFlags, 0, 
		0, NULL, 0, NULL, 1, &imageMemoryBarrier);

    res = vkCreateImageView(device, &color_image_view, NULL, &swapChainViews[i]);
    if (res != VK_SUCCESS) {
      printf ("vkCreateImageView returned error.\n");
      return -1;
    }
    printf ("Colour buffer image view created: %d\n", swapChainViews[i]);
  }  
  printf ("swapchainImageCount %d.\n", swapchainImageCount);
}

int initDepthBuffer(VkPhysicalDevice *physicalDevices, VkDevice device, VkSwapchainKHR swapchain, VkFormat depth_format, VkCommandBuffer setupBuffer, VkImage *pDepthImage, VkImageView *pDepthView, VkDeviceMemory *pDepthMemory)
{  
  VkResult res;
  
  VkImageCreateInfo imageCreateInfo;
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(physicalDevices[0], depth_format, &props);
  if (props.linearTilingFeatures &
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
  } else if (props.optimalTilingFeatures &
	      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
      imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  } else {
      printf ("depth_format %d Unsupported.\n", depth_format);
      return -1;
  }

  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.pNext = NULL;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
  imageCreateInfo.format = depth_format;
  imageCreateInfo.extent.width = width;
  imageCreateInfo.extent.height = height;
  imageCreateInfo.extent.depth = 1;
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = 1;
  imageCreateInfo.samples = NUM_SAMPLES;
  imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.queueFamilyIndexCount = 0;
  imageCreateInfo.pQueueFamilyIndices = NULL;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  imageCreateInfo.flags = 0;

  VkImageViewCreateInfo view_info;
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.pNext = NULL;
  view_info.image = VK_NULL_HANDLE;
  view_info.format = depth_format;
  view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  view_info.components.r = VK_COMPONENT_SWIZZLE_R;
  view_info.components.g = VK_COMPONENT_SWIZZLE_G;
  view_info.components.b = VK_COMPONENT_SWIZZLE_B;
  view_info.components.a = VK_COMPONENT_SWIZZLE_A;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = 1;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.flags = 0;
  
  //Create image for depth buffer
  res = vkCreateImage(device, &imageCreateInfo, NULL, pDepthImage);
  if (res != VK_SUCCESS) {
    printf ("vkCreateImage returned error while creating depth buffer.\n");
    return -1;
  }  
  printf ("Depth image created: %d\n", *pDepthImage);

  VkMemoryRequirements memoryRequirements;
  vkGetImageMemoryRequirements(device, *pDepthImage, &memoryRequirements);  
 
  uint32_t typeBits = memoryRequirements.memoryTypeBits;
  uint32_t typeIndex;
  //Get the index of the first set bit:
  for (typeIndex = 0; typeIndex < 32; typeIndex++) {
    if ((typeBits & 1) == 1)//Check last bit;
      break;
    typeBits >>= 1;
  }
  
  VkMemoryAllocateInfo memAllocInfo;
  memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocInfo.pNext = NULL;
  memAllocInfo.allocationSize = memoryRequirements.size;
  memAllocInfo.memoryTypeIndex = typeIndex;
  
  //Allocate memory
  res = vkAllocateMemory(device, &memAllocInfo, NULL, pDepthMemory);
  if (res != VK_SUCCESS) {
    printf ("vkAllocateMemory returned error while creating depth buffer.\n");
    return -1;
  }  
  printf ("Depth image memory allocated\n");
    
  //Bind memory
  res = vkBindImageMemory(device, *pDepthImage, *pDepthMemory, 0);
  if (res != VK_SUCCESS) {
    printf ("vkBindImageMemory returned error while creating depth buffer. %d\n", res);
    return -1;
  }  
    
  VkImageMemoryBarrier imageMemoryBarrier;
  imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  imageMemoryBarrier.pNext = NULL;
  imageMemoryBarrier.image = *pDepthImage;
  imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
  imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
  imageMemoryBarrier.subresourceRange.levelCount = 1;
  imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
  imageMemoryBarrier.subresourceRange.layerCount = 1;
  imageMemoryBarrier.srcQueueFamilyIndex=0;
  imageMemoryBarrier.dstQueueFamilyIndex=0;
  imageMemoryBarrier.srcAccessMask = 0;
  imageMemoryBarrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

  // Put barrier on top
  VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
  VkPipelineStageFlags destStageFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

  // Put barrier inside setup command buffer
  vkCmdPipelineBarrier(setupBuffer, srcStageFlags, destStageFlags, 0, 
		0, NULL, 0, NULL, 1, &imageMemoryBarrier);

  //Create image view  
  view_info.image = *pDepthImage;
  res = vkCreateImageView(device, &view_info, NULL, pDepthView);
  if (res != VK_SUCCESS) {
    printf ("vkCreateImageView returned error while creating depth buffer. %d\n", res);
    return -1;
  }    
  printf ("Depth image view created: %d\n", *pDepthView);
}

int initFrameBuffers(VkDevice device, VkRenderPass renderPass, int swapchainImageCount, VkImageView *swapChainViews, VkImageView depthView, VkFramebuffer **pFramebuffers)
{  
  int i;
  VkFramebuffer *framebuffers=malloc(sizeof(VkFramebuffer)*swapchainImageCount);
  *pFramebuffers=framebuffers;
  VkResult res;
  
  VkFramebufferCreateInfo fb_info;  
  for (i = 0; i < swapchainImageCount; i++) {
    
    VkImageView imageViewAttachments[2];

    //Attach the correct swapchain colourbuffer
    imageViewAttachments[0] = swapChainViews[i];
    //We only have one depth buffer which we attach to all framebuffers
    imageViewAttachments[1] = depthView;
    
    fb_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fb_info.pNext = NULL;
    fb_info.renderPass = renderPass;
    fb_info.attachmentCount = 2;
    fb_info.pAttachments = imageViewAttachments;
    fb_info.width = width;
    fb_info.height = height;
    fb_info.layers = 1;

    res = vkCreateFramebuffer(device, &fb_info, NULL, &framebuffers[i]);
    if (res != VK_SUCCESS) {
      printf ("vkCreateFramebuffer returned error %d.\n", res);
      return -1;
    }
  }
}

int resizeBuffers(VkCommandBuffer setupBuffer, VkQueue queue, VkPhysicalDevice *physicalDevices, VkDevice device, VkSurfaceKHR surface, VkSwapchainKHR *pSwapchain, VkFormat *pFormat, VkImage **pSwapChainImages, VkImageView **pSwapChainViews, int *pSwapchainImageCount, VkFormat depth_format, VkImage *pDepthImage, VkImageView *pDepthView, VkDeviceMemory *pDepthMemory)
{  
  VkResult res;
  printf("resize() %d, %d\n",width,height);
  
  //First thing we need to do is delete the existing buffers (or we will have a graphics memory leak):
  printf("Destroying depth image view: %d\n", *pDepthView);
  vkDestroyImageView(device, *pDepthView, 0);
  printf("Freeing depth memory\n");
  vkFreeMemory(device, *pDepthMemory, 0);
  printf("Destroying depth image: %d\n", *pDepthImage);
  vkDestroyImage(device, *pDepthImage, 0); 
  for (int i =0; i<*pSwapchainImageCount; i++)
  {
      printf("Destroying swapchain image view: %d\n", (*pSwapChainViews)[i]);
      vkDestroyImageView(device, (*pSwapChainViews)[i], 0);
  }
  //We do not have to free the memory or destroy images for the swapchain (this is done for us by vkDestroySwapchainKHR)
  
  if (initSwapchain(physicalDevices, device, surface, pSwapchain, pFormat)<0)
  return -1;   

  VkCommandBufferBeginInfo commandBufferBeginInfo = {};
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = NULL;
  commandBufferBeginInfo.flags = 0;
  commandBufferBeginInfo.pInheritanceInfo = NULL;
  
  res = vkBeginCommandBuffer(setupBuffer, &commandBufferBeginInfo);
  if (res != VK_SUCCESS) {
    printf ("vkBeginCommandBuffer returned error.\n");
    return -1;
  }

  if (initColourBuffers(physicalDevices, device, *pSwapchain, *pFormat, setupBuffer,  pSwapChainImages, pSwapChainViews, pSwapchainImageCount) <0)
    return -1;
  
  if (initDepthBuffer(physicalDevices, device, *pSwapchain, depth_format, setupBuffer, pDepthImage, pDepthView, pDepthMemory) <0)
    return -1;
  
  res = vkEndCommandBuffer(setupBuffer);
  if (res != VK_SUCCESS) {
    printf ("vkEndCommandBuffer returned error %d.\n", res);
    return -1;
  }
    
  //Submit the setup command buffer
  VkSubmitInfo submitInfo[1];
  submitInfo[0].pNext = NULL;
  submitInfo[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo[0].waitSemaphoreCount = 0;
  submitInfo[0].pWaitSemaphores = NULL;
  submitInfo[0].pWaitDstStageMask = NULL;
  submitInfo[0].commandBufferCount = 1;
  submitInfo[0].pCommandBuffers = &setupBuffer;
  submitInfo[0].signalSemaphoreCount = 0;
  submitInfo[0].pSignalSemaphores = NULL;

  //Queue the command buffer for execution
  res = vkQueueSubmit(queue, 1, submitInfo, VK_NULL_HANDLE);
  if (res != VK_SUCCESS) {
    printf ("vkQueueSubmit returned error %d.\n", res);
    return -1;
  }
  
  res = vkQueueWaitIdle(queue);
  if (res != VK_SUCCESS) {
    printf ("vkQueueWaitIdle returned error %d.\n", res);
    return -1;
  }
}

int main(int argc, char* argv[])
{
  //Initialize the VkApplicationInfo structure
  VkResult res;
  VkApplicationInfo app_info;
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = NULL;
  app_info.pApplicationName = "My Test App";
  app_info.applicationVersion = 1;
  app_info.pEngineName = "My Test App Engine";
  app_info.engineVersion = 1;
  app_info.apiVersion = VK_MAKE_VERSION(1, 0, 3);

  const char *enabledInstanceExtensionNames[] = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_KHR_XCB_SURFACE_EXTENSION_NAME
  };

  //Print a list of available extensions:
  uint32_t avalibleLayerCount=0;
  res = vkEnumerateInstanceLayerProperties(&avalibleLayerCount, NULL);
  printf("There are %d layers avalible\n", avalibleLayerCount);
  VkLayerProperties avalibleLayers[avalibleLayerCount];
  res = vkEnumerateInstanceLayerProperties(&avalibleLayerCount, avalibleLayers);
  for (int i =0; i<avalibleLayerCount; i++)
  {     
    printf("%s: %s\n", avalibleLayers[i].layerName, avalibleLayers[i].description);
  }

  const char *enabledLayerNames[] = {
    //List any layers you want to enable here.
  };
  uint32_t enabledLayerCount=0;

  //Initialize the VkInstanceCreateInfo structure
  VkInstanceCreateInfo inst_info;
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pNext = NULL;
  inst_info.flags = 0;
  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledExtensionCount = 2;
  inst_info.ppEnabledExtensionNames = enabledInstanceExtensionNames;
  inst_info.enabledLayerCount = enabledLayerCount;
  inst_info.ppEnabledLayerNames = enabledLayerNames;

  VkInstance inst;

  res = vkCreateInstance(&inst_info, NULL, &inst);
  if (res == VK_ERROR_INCOMPATIBLE_DRIVER) {	    
    printf ("vkCreateInstance returned VK_ERROR_INCOMPATIBLE_DRIVER\n");
    return -1;
  } else if (res != VK_SUCCESS) {
    printf ("vkCreateInstance returned error %d\n", res);
    return -1;
  }

  printf ("Vulkan working\n");

  //Setup XCB Connection:
  const xcb_setup_t *setup;
  xcb_screen_iterator_t iter;
  int scr;

  xcb_connection_t *connection = xcb_connect(NULL, &scr);
  if (connection == NULL) {
    printf("Cannot find a compatible Vulkan ICD.\n");
    return -1;
  }

  setup = xcb_get_setup(connection);
  iter = xcb_setup_roots_iterator(setup);
  while (scr-- > 0)
    xcb_screen_next(&iter);
  xcb_screen_t *screen = iter.data;
  
  //Create an xcb window:
  uint32_t value_mask, value_list[32], window;
  
  window = xcb_generate_id(connection);

  //value_mask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;  
  value_mask =  XCB_CW_EVENT_MASK;
  //value_list[0] = screen->black_pixel;
  value_list[0] = XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_BUTTON_PRESS | XCB_EVENT_MASK_STRUCTURE_NOTIFY;

  xcb_create_window(connection, XCB_COPY_FROM_PARENT, window,
                    screen->root, 0, 0, width, height, 0,
                    XCB_WINDOW_CLASS_INPUT_OUTPUT, screen->root_visual,
                    value_mask, value_list);

  //We want to know when the user presses the close button:
  xcb_intern_atom_cookie_t cookie = xcb_intern_atom(connection, 1, 12, "WM_PROTOCOLS");
  xcb_intern_atom_reply_t* reply = xcb_intern_atom_reply(connection, cookie, 0);

  xcb_intern_atom_cookie_t cookie2 = xcb_intern_atom(connection, 0, 16, "WM_DELETE_WINDOW");
  xcb_intern_atom_reply_t* delete_window_reply = xcb_intern_atom_reply(connection, cookie2, 0);

  xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, (*reply).atom, 4, 32, 1, &(*delete_window_reply).atom);
  char* windowTitle="Vulkan Example";
  xcb_change_property(connection, XCB_PROP_MODE_REPLACE, window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, strlen(windowTitle), windowTitle); 
  
  xcb_map_window(connection, window);
    
  xcb_flush(connection); 
  
    //Wait until the window has been exposed:
  xcb_generic_event_t *e;
  while ((e = xcb_wait_for_event(connection))) {
    if ((e->response_type & ~0x80) == XCB_EXPOSE)
      break;
  }

  VkSurfaceKHR surface;
  VkXcbSurfaceCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR;
  createInfo.pNext = NULL;
  createInfo.connection = connection;
  createInfo.window = window;
  res = vkCreateXcbSurfaceKHR(inst, &createInfo, NULL, &surface);
  if (res != VK_SUCCESS) {
    printf ("vkCreateXcbSurfaceKHR returned error.\n");
    return -1;
  }  VkSurfaceCapabilitiesKHR surfaceCapabilities;

  uint32_t deviceBufferSize=0;
  res = vkEnumeratePhysicalDevices(inst, &deviceBufferSize, NULL);
  printf ("GPU Count: %i\n", deviceBufferSize);
  VkPhysicalDevice physicalDevices[deviceBufferSize];
  res = vkEnumeratePhysicalDevices(inst, &deviceBufferSize, physicalDevices);
  if (res == VK_ERROR_INITIALIZATION_FAILED) {
    printf ("vkEnumeratePhysicalDevices returned VK_ERROR_INITIALIZATION_FAILED for GPU 0.\n");
    return -1;
  }else if (res != VK_SUCCESS) {
    printf ("vkEnumeratePhysicalDevices returned error.\n");
    return -1;
  }

  VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &physicalDeviceMemoryProperties);
  printf ("There are %d memory types.\n", physicalDeviceMemoryProperties.memoryTypeCount);
  
  VkDeviceQueueCreateInfo deviceQueueCreateInfo;
  deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  deviceQueueCreateInfo.pNext = NULL;
  deviceQueueCreateInfo.queueCount = 1;
  float queuePriorities[1] = {1.0};
  deviceQueueCreateInfo.pQueuePriorities = queuePriorities;
  
  uint32_t queueCount=0;
  //We are only using the first physical device:
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueCount, NULL);  
  printf ("%i PhysicalDeviceQueueFamily(ies).\n", queueCount);
  
  VkQueueFamilyProperties queueFamilyProperties[queueCount];
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueCount, queueFamilyProperties);
  int found = 0;
  unsigned int i = 0;
  VkBool32 supportsPresent;
  for (; i < queueCount; i++) {
    if (queueFamilyProperties[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      printf ("PhysicalDeviceQueueFamily %i has property VK_QUEUE_GRAPHICS_BIT.\n", i);
      vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevices[0], i, surface, &supportsPresent);
      if (supportsPresent) {
        deviceQueueCreateInfo.queueFamilyIndex = i;
        found = 1;
        break;
      }
    }
  }  
  if (found==0) {
    printf ("Error: A suitable queue family has not been found.\n");
    return -1;
  }

  const char *enabledDeviceExtensionNames[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
  };

  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pNext = NULL;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &deviceQueueCreateInfo;
  dci.enabledLayerCount = 0;
  dci.ppEnabledLayerNames = NULL;
  dci.enabledExtensionCount = 1;
  dci.ppEnabledExtensionNames = enabledDeviceExtensionNames;
  dci.pEnabledFeatures = NULL;
  dci.enabledLayerCount = enabledLayerCount;
  dci.ppEnabledLayerNames = enabledLayerNames;

  VkDevice device;
  res = vkCreateDevice(physicalDevices[0], &dci, NULL, &device);
  if (res == VK_ERROR_INITIALIZATION_FAILED) {
    printf ("vkCreateDevice returned VK_ERROR_INITIALIZATION_FAILED for GPU 0.\n");
    return -1;
  }else if (res != VK_SUCCESS) {
    printf ("vkCreateDevice returned error %d.\n", res);
    return -1;
  }  
  
  
  //Setup Command buffers
  VkCommandPool commandPool;
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = NULL;
  commandPoolCreateInfo.queueFamilyIndex = deviceQueueCreateInfo.queueFamilyIndex;
  commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  res = vkCreateCommandPool(device, &commandPoolCreateInfo, NULL, &commandPool);
  if (res != VK_SUCCESS) {
    printf ("vkCreateCommandPool returned error.\n");
    return -1;
  }

  VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
  commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  commandBufferAllocateInfo.pNext = NULL;
  commandBufferAllocateInfo.commandPool = commandPool;
  commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  commandBufferAllocateInfo.commandBufferCount = 2;

  VkCommandBuffer commandBuffers[2];
  res = vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers);
  if (res != VK_SUCCESS) {
    printf ("vkAllocateCommandBuffers returned error.\n");
    return -1;
  }

  VkQueue queue;
  vkGetDeviceQueue(device, deviceQueueCreateInfo.queueFamilyIndex, 0, &queue);

  VkSemaphore presentCompleteSemaphore;
  VkSemaphoreCreateInfo presentCompleteSemaphoreCreateInfo;
  presentCompleteSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  presentCompleteSemaphoreCreateInfo.pNext = NULL;
  presentCompleteSemaphoreCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

  res = vkCreateSemaphore(device, &presentCompleteSemaphoreCreateInfo, NULL, &presentCompleteSemaphore);
  if (res != VK_SUCCESS) {
    printf ("vkCreateSemaphore returned error.\n");
    return -1;
  }
  
  VkCommandBufferBeginInfo commandBufferBeginInfo = {};
  commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  commandBufferBeginInfo.pNext = NULL;
  commandBufferBeginInfo.flags = 0;
  commandBufferBeginInfo.pInheritanceInfo = NULL;
  res = vkBeginCommandBuffer(commandBuffers[0], &commandBufferBeginInfo);
  if (res != VK_SUCCESS) {
    printf ("vkBeginCommandBuffer returned error.\n");
    return -1;
  }
      
  VkFormat format;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  if (initSwapchain(physicalDevices, device, surface, &swapchain, &format)<0)
    return -1;
  
  VkImage *swapChainImages;
  VkImageView *swapChainViews;
  
  uint32_t swapchainImageCount = 0;  
  if (initColourBuffers(physicalDevices, device, swapchain, format, commandBuffers[0],  &swapChainImages, &swapChainViews, &swapchainImageCount) <0)
    return -1;
  /*
  if (depth_format == VK_FORMAT_D16_UNORM_S8_UINT ||
      depth_format == VK_FORMAT_D24_UNORM_S8_UINT ||
      depth_format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
      view_info.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
  }*/
  
//Setup the depth buffer:
  const VkFormat depth_format = VK_FORMAT_D16_UNORM;
  VkImage depthImage;
  VkImageView depthView;  
  VkDeviceMemory depthMemory;
  initDepthBuffer(physicalDevices, device, swapchain, depth_format, commandBuffers[0], &depthImage, &depthView, &depthMemory);
		
  res = vkEndCommandBuffer(commandBuffers[0]);
  if (res != VK_SUCCESS) {
    printf ("vkEndCommandBuffer returned error %d.\n", res);
    return -1;
  }  

  //Submit the setup command buffer
  VkSubmitInfo submitInfo[1];
  submitInfo[0].pNext = NULL;
  submitInfo[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo[0].waitSemaphoreCount = 0;
  submitInfo[0].pWaitSemaphores = NULL;
  submitInfo[0].pWaitDstStageMask = NULL;
  submitInfo[0].commandBufferCount = 1;
  submitInfo[0].pCommandBuffers = &commandBuffers[0];
  submitInfo[0].signalSemaphoreCount = 0;
  submitInfo[0].pSignalSemaphores = NULL;

  //Queue the command buffer for execution
  res = vkQueueSubmit(queue, 1, submitInfo, VK_NULL_HANDLE);
  if (res != VK_SUCCESS) {
    printf ("vkQueueSubmit returned error %d.\n", res);
    return -1;
  }

  res = vkQueueWaitIdle(queue);
  if (res != VK_SUCCESS) {
    printf ("vkQueueWaitIdle returned error %d.\n", res);
    return -1;
  }

  //Setup the renderpass:
  VkAttachmentDescription attachments[2];
  attachments[0].format = format;
  attachments[0].samples = NUM_SAMPLES;
  attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  attachments[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  attachments[0].flags = 0;

  //Depth buffer
  attachments[1].format = depth_format;
  attachments[1].samples = NUM_SAMPLES;
  attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
  attachments[1].initialLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].finalLayout =
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  attachments[1].flags = 0;

  VkAttachmentReference color_reference;
  color_reference.attachment = 0;
  color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depth_reference;
  depth_reference.attachment = 1;
  depth_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass;
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.flags = 0;
  subpass.inputAttachmentCount = 0;
  subpass.pInputAttachments = NULL;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_reference;
  subpass.pResolveAttachments = NULL;
  subpass.pDepthStencilAttachment = &depth_reference;
  subpass.preserveAttachmentCount = 0;
  subpass.pPreserveAttachments = NULL;

  VkRenderPassCreateInfo rp_info;
  rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  rp_info.pNext = NULL;
  rp_info.attachmentCount = 2;
  rp_info.pAttachments = attachments;
  rp_info.subpassCount = 1;
  rp_info.pSubpasses = &subpass;
  rp_info.dependencyCount = 0;
  rp_info.pDependencies = NULL;

  VkRenderPass renderPass;
  res = vkCreateRenderPass(device, &rp_info, NULL, &renderPass);
    if (res != VK_SUCCESS) {
    printf ("vkCreateRenderPass returned error. %d\n", res);
    return -1;
  }    
  
  //Setup the pipeline
  VkDescriptorSetLayoutBinding layout_bindings[1];
  layout_bindings[0].binding = 0;
  layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  layout_bindings[0].descriptorCount = 1;
  layout_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  layout_bindings[0].pImmutableSamplers = NULL;
  
  //Next take layout bindings and use them to create a descriptor set layout
  VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
  descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptorSetLayoutCreateInfo.pNext = NULL;
  descriptorSetLayoutCreateInfo.bindingCount = 1;
  descriptorSetLayoutCreateInfo.pBindings = layout_bindings;
    
  VkDescriptorSetLayout descriptorSetLayout;
  res = vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayout);
  if (res != VK_SUCCESS) {
    printf ("vkCreateDescriptorSetLayout returned error.\n");
    return -1;
  }
  
  //Now use the descriptor layout to create a pipeline layout 
  VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo;
  pPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pPipelineLayoutCreateInfo.pNext = NULL;
  pPipelineLayoutCreateInfo.pushConstantRangeCount = 0;		
  pPipelineLayoutCreateInfo.pPushConstantRanges = NULL;
  pPipelineLayoutCreateInfo.setLayoutCount = 1;
  pPipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;

  VkPipelineLayout pipelineLayout;

  res = vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, NULL, &pipelineLayout);
  if (res != VK_SUCCESS) {
    printf ("vkCreatePipelineLayout returned error.\n");
    return -1;
  }  

  //load shaders    
  size_t vertexShaderSize=0;
  char *vertexShader = readBinaryFile("shaders/vert.spv", &vertexShaderSize);
  size_t fragmentShaderSize=0;
  char *fragmentShader = readBinaryFile("shaders/frag.spv", &fragmentShaderSize);
  if (vertexShaderSize==0 || fragmentShaderSize==0){
    printf ("Colud not load shader file.\n");
    return -1;
  }    

  VkShaderModuleCreateInfo moduleCreateInfo;
  VkPipelineShaderStageCreateInfo shaderStages[2];

  shaderStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStages[0].pNext = NULL;
  shaderStages[0].pSpecializationInfo = NULL;
  shaderStages[0].flags = 0;
  shaderStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  shaderStages[0].pName = "main";

  moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleCreateInfo.pNext = NULL;
  moduleCreateInfo.flags = 0;
  moduleCreateInfo.codeSize = vertexShaderSize;
  moduleCreateInfo.pCode = (uint32_t*)vertexShader; //This may not work with big-endian systems.
  res = vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderStages[0].module);
  if (res != VK_SUCCESS) {
    printf ("vkCreateShaderModule returned error %d.\n", res);
    return -1;
  }  

  shaderStages[1].sType =  VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStages[1].pNext = NULL;
  shaderStages[1].pSpecializationInfo = NULL;
  shaderStages[1].flags = 0;
  shaderStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  shaderStages[1].pName = "main";

  moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  moduleCreateInfo.pNext = NULL;
  moduleCreateInfo.flags = 0;
  moduleCreateInfo.codeSize = fragmentShaderSize;
  moduleCreateInfo.pCode = (uint32_t*)fragmentShader;
  res = vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderStages[1].module);
  if (res != VK_SUCCESS) {
    printf ("vkCreateShaderModule returned error %d.\n", res);
    return -1;
  }
  
  //Create the framebuffers
  VkFramebuffer *framebuffers=0;
  if (initFrameBuffers(device, renderPass, swapchainImageCount, swapChainViews, depthView, &framebuffers)<0)
    return -1;
  
  //Create the uniforms
  float projectionMatrix[16];
  float viewMatrix[16];
  float modelMatrix[16]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};  
  float MVMatrix[16];
  float MVPMatrix[16];
  
  perspective_matrix(0.7853 /* 45deg */, (float)width/(float)height, 0.1f, 100.0f, projectionMatrix);
  translate_matrix(0,0,-5, viewMatrix);
  rotate_matrix(45, 0,1,0, modelMatrix);
  multiply_matrix(viewMatrix, modelMatrix, MVMatrix);
  multiply_matrix(projectionMatrix, MVMatrix, MVPMatrix);
  /*
  ("viewMatrix\n");
  print_matrix(viewMatrix);
  printf("projectionMatrix\n");
  print_matrix(projectionMatrix);
  printf("MVPMatrix\n");
  print_matrix(MVPMatrix);
  */

  VkBufferCreateInfo uniformBufferCreateInfo;
  uniformBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  uniformBufferCreateInfo.pNext = NULL;
  uniformBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  uniformBufferCreateInfo.size = sizeof(MVPMatrix);
  uniformBufferCreateInfo.queueFamilyIndexCount = 0;
  uniformBufferCreateInfo.pQueueFamilyIndices = NULL;
  uniformBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  uniformBufferCreateInfo.flags = 0;
  
  VkBuffer uniformBuffer;
  res = vkCreateBuffer(device, &uniformBufferCreateInfo, NULL, &uniformBuffer);
  if (res != VK_SUCCESS) {
    printf ("vkCreateBuffer returned error %d.\n", res);
    return -1;
  }

  VkMemoryRequirements memoryRequirements;
  vkGetBufferMemoryRequirements(device, uniformBuffer, &memoryRequirements);
  
  uint32_t typeBits = memoryRequirements.memoryTypeBits;  
  uint32_t typeIndex;
  VkFlags requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;// | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (typeIndex = 0; typeIndex < physicalDeviceMemoryProperties.memoryTypeCount; typeIndex++) {
    if ((typeBits & 1) == 1)//Check last bit;
    {
      if ((physicalDeviceMemoryProperties.memoryTypes[typeIndex].propertyFlags & requirements_mask) == requirements_mask) 
      {
	found=1;
	break;
      }
      typeBits >>= 1;
    }
  }
  
  if (!found)
  {
    printf ("Did not find a suitible memory type.\n");
    return -1;
  }else    
    printf ("Using memory type %d.\n", typeIndex);
  
  VkMemoryAllocateInfo memAllocInfo;
  memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocInfo.pNext = NULL;
  memAllocInfo.allocationSize = memoryRequirements.size;
  memAllocInfo.memoryTypeIndex = typeIndex;
// 
  VkDeviceMemory uniformMemory;
  res = vkAllocateMemory(device, &memAllocInfo, NULL, &uniformMemory);    
  if (res != VK_SUCCESS) {
    printf ("vkCreateBuffer returned error %d.\n", res);
    return -1;
  }
  
  uint8_t *uniformMappedMemory;
  res = vkMapMemory(device, uniformMemory, 0, memoryRequirements.size, 0, (void **)&uniformMappedMemory);
  if (res != VK_SUCCESS) {
    printf ("vkMapMemory returned error %d.\n", res);
    return -1;
  }
   
  memcpy(uniformMappedMemory, MVPMatrix, sizeof(MVPMatrix));

  //vkUnmapMemory(device, uniformMemory);

  res = vkBindBufferMemory(device, uniformBuffer, uniformMemory, 0);
  if (res != VK_SUCCESS) {
    printf ("vkBindBufferMemory returned error %d.\n", res);
    return -1;
  }
  
  VkDescriptorBufferInfo uniformBufferInfo;
  uniformBufferInfo.buffer = uniformBuffer;
  uniformBufferInfo.offset = 0;
  uniformBufferInfo.range = sizeof(MVPMatrix);
  
  //Create a descriptor pool
  VkDescriptorPoolSize typeCounts[1];
  typeCounts[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  typeCounts[0].descriptorCount = 1;

  VkDescriptorPoolCreateInfo descriptorPoolInfo;
  descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolInfo.pNext = NULL;
  descriptorPoolInfo.maxSets = 1;
  descriptorPoolInfo.poolSizeCount = 1;
  descriptorPoolInfo.pPoolSizes = typeCounts;

  VkDescriptorPool descriptorPool;
  res = vkCreateDescriptorPool(device, &descriptorPoolInfo, NULL, &descriptorPool);
  if (res != VK_SUCCESS) {
    printf ("vkCreateDescriptorPool returned error %d.\n", res);
    return -1;
  } 
 
  //Create Vertex buffers:
  VkBufferCreateInfo vertexBufferCreateInfo;
  vertexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  vertexBufferCreateInfo.pNext = NULL;
  vertexBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  vertexBufferCreateInfo.size = sizeof(vertexData);
  vertexBufferCreateInfo.queueFamilyIndexCount = 0;
  vertexBufferCreateInfo.pQueueFamilyIndices = NULL;
  vertexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vertexBufferCreateInfo.flags = 0;
  
  VkBuffer vertexBuffer;
  res = vkCreateBuffer(device, &vertexBufferCreateInfo, NULL, &vertexBuffer);
  if (res != VK_SUCCESS) {
    printf ("vkCreateDescriptorPool returned error %d.\n", res);
    return -1;
  }
  
  vkGetBufferMemoryRequirements(device, vertexBuffer, &memoryRequirements);
  typeBits = memoryRequirements.memoryTypeBits;
  requirements_mask = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (typeIndex = 0; typeIndex < physicalDeviceMemoryProperties.memoryTypeCount; typeIndex++) {
    if ((typeBits & 1) == 1)//Check last bit;
    {
      if ((physicalDeviceMemoryProperties.memoryTypes[typeIndex].propertyFlags & requirements_mask) == requirements_mask) 
      {
	found=1;
	break;
      }
      typeBits >>= 1;
    }
  }
  
  if (!found)
  {
    printf ("Did not find a suitible memory type.\n");
    return -1;
  }else    
    printf ("Using memory type %d.\n", typeIndex);
  
  memAllocInfo.pNext = NULL;
  memAllocInfo.allocationSize = memoryRequirements.size;
  memAllocInfo.memoryTypeIndex = typeIndex;
  
  VkDeviceMemory vertexMemory;
  res = vkAllocateMemory(device, &memAllocInfo, NULL, &vertexMemory);
  if (res != VK_SUCCESS) {
    printf ("vkAllocateMemory returned error %d.\n", res);
    return -1;
  }

  uint8_t *vertexMappedMemory;
  res = vkMapMemory(device, vertexMemory, 0, memoryRequirements.size, 0, (void **)&vertexMappedMemory);
  if (res != VK_SUCCESS) {
    printf ("vkMapMemory returned error %d.\n", res);
    return -1;
  }
  
  memcpy(vertexMappedMemory, vertexData, sizeof(vertexData));

  vkUnmapMemory(device, vertexMemory);

  res = vkBindBufferMemory(device, vertexBuffer, vertexMemory, 0);
  if (res != VK_SUCCESS) {
    printf ("vkBindBufferMemory returned error %d.\n", res);
    return -1;
  }
  VkVertexInputBindingDescription vertexInputBindingDescription;
  vertexInputBindingDescription.binding = 0;
  vertexInputBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  vertexInputBindingDescription.stride = sizeof(vertexData[0]);

  VkVertexInputAttributeDescription vertexInputAttributeDescription[2];
  vertexInputAttributeDescription[0].binding = 0;
  vertexInputAttributeDescription[0].location = 0;
  vertexInputAttributeDescription[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  vertexInputAttributeDescription[0].offset = 0;
  vertexInputAttributeDescription[1].binding = 0;
  vertexInputAttributeDescription[1].location = 1;
  vertexInputAttributeDescription[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  vertexInputAttributeDescription[1].offset = 16;

  //Create a descriptor set
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
  descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = NULL;
  descriptorSetAllocateInfo.descriptorPool = descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount = 1;
  descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

  //return 0;
  VkDescriptorSet descriptorSets[1];
  res = vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, descriptorSets);
  if (res != VK_SUCCESS) {
    printf ("vkAllocateDescriptorSets returned error %d.\n", res);
    return -1;
  }
  VkWriteDescriptorSet writes[1];
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].pNext = NULL;
  writes[0].dstSet = descriptorSets[0];
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  writes[0].pBufferInfo = &uniformBufferInfo;
  writes[0].dstArrayElement = 0;
  writes[0].dstBinding = 0;

  vkUpdateDescriptorSets(device, 1, writes, 0, NULL); 
    /*
  //Create a pipeline cache
  VkPipelineCache pipelineCache;
  VkPipelineCacheCreateInfo pipelineCacheInfo;
  pipelineCacheInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  pipelineCacheInfo.pNext = NULL;
  pipelineCacheInfo.initialDataSize = 0;
  pipelineCacheInfo.pInitialData = NULL;
  pipelineCacheInfo.flags = 0;
  res = vkCreatePipelineCache(device, &pipelineCacheInfo, NULL, &pipelineCache);
  if (res != VK_SUCCESS) {
    printf ("vkCreatePipelineCache returned error %d.\n", res);
    return -1;
  }  */
      
  //Create a pipeline object
  VkDynamicState dynamicStateEnables[VK_DYNAMIC_STATE_RANGE_SIZE];
  VkPipelineDynamicStateCreateInfo dynamicState;
  //No dynamic state:
  memset(dynamicStateEnables, 0, sizeof dynamicStateEnables);
  dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.pNext = NULL;
  dynamicState.pDynamicStates = dynamicStateEnables;
  dynamicState.dynamicStateCount = 0;

  VkPipelineVertexInputStateCreateInfo vi;
  vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vi.pNext = NULL;
  vi.flags = 0;
  vi.vertexBindingDescriptionCount = 1;
  vi.pVertexBindingDescriptions = &vertexInputBindingDescription;
  vi.vertexAttributeDescriptionCount = 2;
  vi.pVertexAttributeDescriptions = vertexInputAttributeDescription;

  VkPipelineInputAssemblyStateCreateInfo ia;
  ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  ia.pNext = NULL;
  ia.flags = 0;
  ia.primitiveRestartEnable = VK_FALSE;
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  VkPipelineRasterizationStateCreateInfo rs;
  rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rs.pNext = NULL;
  rs.flags = 0;
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_BACK_BIT;
  rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rs.depthClampEnable = VK_TRUE;
  rs.rasterizerDiscardEnable = VK_FALSE;
  rs.depthBiasEnable = VK_FALSE;
  rs.depthBiasConstantFactor = 0;
  rs.depthBiasClamp = 0;
  rs.depthBiasSlopeFactor = 0;
  rs.lineWidth = 1;

  VkPipelineColorBlendStateCreateInfo cb;
  cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  cb.flags = 0;
  cb.pNext = NULL;
  VkPipelineColorBlendAttachmentState att_state[1];
  att_state[0].colorWriteMask = 0xf;
  att_state[0].blendEnable = VK_FALSE;
  att_state[0].alphaBlendOp = VK_BLEND_OP_ADD;
  att_state[0].colorBlendOp = VK_BLEND_OP_ADD;
  att_state[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  att_state[0].dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
  att_state[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  att_state[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  cb.attachmentCount = 1;
  cb.pAttachments = att_state;
  cb.logicOpEnable = VK_FALSE;
  cb.logicOp = VK_LOGIC_OP_NO_OP;
  cb.blendConstants[0] = 1.0f;
  cb.blendConstants[1] = 1.0f;
  cb.blendConstants[2] = 1.0f;
  cb.blendConstants[3] = 1.0f;

  VkPipelineViewportStateCreateInfo vp = {};
  vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  vp.pNext = NULL;
  vp.flags = 0;
  vp.viewportCount = 1;
  dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_VIEWPORT;
  vp.scissorCount = 1;
  dynamicStateEnables[dynamicState.dynamicStateCount++] = VK_DYNAMIC_STATE_SCISSOR;
  vp.pScissors = NULL;
  vp.pViewports = NULL;

  VkPipelineDepthStencilStateCreateInfo ds;
  ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  ds.pNext = NULL;
  ds.flags = 0;
  ds.depthTestEnable = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  ds.depthBoundsTestEnable = VK_FALSE;
  ds.stencilTestEnable = VK_FALSE;
  ds.back.failOp = VK_STENCIL_OP_KEEP;
  ds.back.passOp = VK_STENCIL_OP_KEEP;
  ds.back.compareOp = VK_COMPARE_OP_ALWAYS;
  ds.back.compareMask = 0;
  ds.back.reference = 0;
  ds.back.depthFailOp = VK_STENCIL_OP_KEEP;
  ds.back.writeMask = 0;
  ds.minDepthBounds = 0;
  ds.maxDepthBounds = 0;
  ds.stencilTestEnable = VK_FALSE;
  ds.front = ds.back;

  VkPipelineMultisampleStateCreateInfo ms;
  ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  ms.pNext = NULL;
  ms.flags = 0;
  ms.pSampleMask = NULL;
  ms.rasterizationSamples = NUM_SAMPLES;
  ms.sampleShadingEnable = VK_FALSE;
  ms.alphaToCoverageEnable = VK_FALSE;
  ms.alphaToOneEnable = VK_FALSE;
  ms.minSampleShading = 0.0;

  VkGraphicsPipelineCreateInfo pipelineInfo;
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.pNext = NULL;
  pipelineInfo.layout = pipelineLayout;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  pipelineInfo.basePipelineIndex = 0;
  pipelineInfo.flags = 0;
  pipelineInfo.pVertexInputState = &vi;
  pipelineInfo.pInputAssemblyState = &ia;
  pipelineInfo.pRasterizationState = &rs;
  pipelineInfo.pColorBlendState = &cb;
  pipelineInfo.pTessellationState = NULL;
  pipelineInfo.pMultisampleState = &ms;
  pipelineInfo.pDynamicState = &dynamicState;
  pipelineInfo.pViewportState = &vp;
  pipelineInfo.pDepthStencilState = &ds;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.stageCount = 2;
  pipelineInfo.renderPass = renderPass;
  pipelineInfo.subpass = 0;
  
  VkPipeline pipeline;  
  res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &pipeline);
  if (res != VK_SUCCESS) {
    printf ("vkCreateGraphicsPipelines re\turned error %d.\n", res);
    return -1;
  }   

  VkClearValue clear_values[2];
  clear_values[0].color.float32[0] = 0.2f;
  clear_values[0].color.float32[1] = 0.2f;
  clear_values[0].color.float32[2] = 0.2f;
  clear_values[0].color.float32[3] = 0.2f;
  clear_values[1].depthStencil.depth = 1.0f;
  clear_values[1].depthStencil.stencil = 0;

  presentCompleteSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  presentCompleteSemaphoreCreateInfo.pNext = NULL;
  presentCompleteSemaphoreCreateInfo.flags = 0;
  
  uint32_t frame=0;
  int done = 0;
  int newSwapchainRequired = 0;
  uint32_t currentBuffer;

  //The main event loop
  while (1==1) {    
    //printf ("Starting frame %d.\n", frame);
   
    //This semaphore will be signalled after the next image is acquired. The first command buffer will wait until this happens before starting the render pass.
    //Acquire the next image in the swapchain
    res = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, presentCompleteSemaphore, NULL, &currentBuffer);
    if (!(res == VK_SUCCESS || res == VK_ERROR_OUT_OF_DATE_KHR)) {
      printf ("vkAcquireNextImageKHR returned error.\n");
      return -1;
    }else if (res == VK_ERROR_OUT_OF_DATE_KHR)
    {
        printf ("vkAcquireNextImageKHR returned VK_ERROR_OUT_OF_DATE_KHR.\n");
        newSwapchainRequired=1;
    }

    while (1==1) {
      e = xcb_poll_for_event(connection);
      if (!e) break;
      switch(e->response_type & ~0x80)
      {
        case XCB_EXPOSE:
           break;
        case XCB_EVENT_MASK_BUTTON_PRESS:
           //done=1;
           break;
	      case XCB_KEY_PRESS:	  
	        done=1;
                 break;	   
	      case XCB_CONFIGURE_NOTIFY:;
              const xcb_configure_notify_event_t *cfg = (const xcb_configure_notify_event_t *)e;
              if ((width != cfg->width) || (height != cfg->height)) {
                  //The window has been re-sized
                  printf ("Window resized.\n");
                  newSwapchainRequired=1;
                  width = cfg->width;
                  height = cfg->height;
                  //The aspect ratio may have changed, build a now perspective matrix:
                  perspective_matrix(0.7853 /* 45deg */, (float)width/(float)height, 0.1f, 100.0f, projectionMatrix);
                  multiply_matrix(projectionMatrix, MVMatrix, (float*)uniformMappedMemory);
              }
          break;
      }
      if ((e->response_type & ~0x80)==XCB_CLIENT_MESSAGE)
      {
        printf("XCB_CLIENT_MESSAGE");
        if(((xcb_client_message_event_t*)e)->data.data32[0] == delete_window_reply->atom)
          done=1;  
      }
      free(e);
    }
    if (done)
		  break;

    if (newSwapchainRequired)
    {
      newSwapchainRequired=0;
      //Delete the existing framebuffers:
      for (i = 0; i < swapchainImageCount; i++)
        vkDestroyFramebuffer(device, framebuffers[i], 0);
      free(framebuffers);
      //Rebuild the swapchain:
      if (resizeBuffers(commandBuffers[0], queue, physicalDevices, device, surface, &swapchain, &format, &swapChainImages, &swapChainViews, &swapchainImageCount, depth_format, &depthImage, &depthView, &depthMemory)<0)
        return -1;
      //Create new framebuffers:
   	  if (initFrameBuffers(device, renderPass, swapchainImageCount, swapChainViews, depthView, &framebuffers)<0)
        return -1;
      //We must acquire an image from the new swapchain 
      vkDestroySemaphore(device, presentCompleteSemaphore, NULL);
      res = vkCreateSemaphore(device, &presentCompleteSemaphoreCreateInfo, NULL,
              &presentCompleteSemaphore);
      if (res != VK_SUCCESS) {
        printf ("vkCreateSemaphore returned error %d.\n", res);
        return -1;
      }

      res = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, presentCompleteSemaphore, 0, &currentBuffer);
      if (res != VK_SUCCESS) {
        printf ("vkAcquireNextImageKHR returned error.\n");
        return -1;
      }
      fflush(stdout);
    } 
       
    VkRenderPassBeginInfo renderPassBeginInfo;
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.pNext = NULL;
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.framebuffer = framebuffers[currentBuffer];
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clear_values;
    
    res = vkBeginCommandBuffer(commandBuffers[1], &commandBufferBeginInfo);
    if (res != VK_SUCCESS) {
      printf ("vkBeginCommandBuffer returned error.\n");
      return -1;
    }
    
    VkImageMemoryBarrier imageMemoryBarrier;
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    imageMemoryBarrier.pNext = NULL;
    imageMemoryBarrier.image = swapChainImages[currentBuffer];
    imageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    imageMemoryBarrier.subresourceRange.levelCount = 1;
    imageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    imageMemoryBarrier.subresourceRange.layerCount = 1;
    imageMemoryBarrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.srcAccessMask = 0;
    imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    // Put barrier on top
    VkPipelineStageFlags srcStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkPipelineStageFlags destStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    vkCmdPipelineBarrier(commandBuffers[1], srcStageFlags, destStageFlags, 0,
        0, NULL, 0, NULL, 1, &imageMemoryBarrier);
    
    vkCmdBeginRenderPass(commandBuffers[1], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(commandBuffers[1], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(commandBuffers[1], VK_PIPELINE_BIND_POINT_GRAPHICS,
			      pipelineLayout, 0, 1,
			      descriptorSets, 0, NULL);
    
    VkViewport viewport;
    viewport.height = (float)height;
    viewport.width = (float)width;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    viewport.x = 0;
    viewport.y = 0;

    VkRect2D scissor;
    scissor.extent.width = width;
    scissor.extent.height = height;
    scissor.offset.x = 0;
    scissor.offset.y = 0;
    
    vkCmdSetViewport(commandBuffers[1], 0, 1, &viewport);
    vkCmdSetScissor(commandBuffers[1], 0, 1, &scissor);
    
    VkDeviceSize offsets[1] = {0};
    vkCmdBindVertexBuffers(commandBuffers[1], 0, 1, &vertexBuffer, offsets);
    vkCmdDraw(commandBuffers[1], 12 * 3, 1, 0, 0);
    
    vkCmdEndRenderPass(commandBuffers[1]);
    
    VkImageMemoryBarrier prePresentBarrier;
    prePresentBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    prePresentBarrier.pNext = NULL;
    prePresentBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    prePresentBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    prePresentBarrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    prePresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    prePresentBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    prePresentBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    prePresentBarrier.subresourceRange.baseMipLevel = 0;
    prePresentBarrier.subresourceRange.levelCount = 1;
    prePresentBarrier.subresourceRange.baseArrayLayer = 0;
    prePresentBarrier.subresourceRange.layerCount = 1;
    prePresentBarrier.image = swapChainImages[currentBuffer];
    vkCmdPipelineBarrier(commandBuffers[1], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
              VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, NULL, 0,
              NULL, 1, &prePresentBarrier);
    
    res = vkEndCommandBuffer(commandBuffers[1]);
    if (res != VK_SUCCESS) {
      printf ("vkEndCommandBuffer returned error %d.\n", res);
      return -1;
    }    
    
    VkFenceCreateInfo fenceInfo;
    VkFence drawFence;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = NULL;
    fenceInfo.flags = 0;
    vkCreateFence(device, &fenceInfo, NULL, &drawFence);
  
    VkPipelineStageFlags pipe_stage_flags = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    
    //Submit the main render command buffer
    VkSubmitInfo submitInfo[1];
    submitInfo[0].pNext = NULL;
    submitInfo[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo[0].waitSemaphoreCount = 1;
    submitInfo[0].pWaitSemaphores = &presentCompleteSemaphore;
    submitInfo[0].pWaitDstStageMask = &pipe_stage_flags;
    submitInfo[0].commandBufferCount = 1;
    submitInfo[0].pCommandBuffers = &commandBuffers[1];
    submitInfo[0].signalSemaphoreCount = 0;
    submitInfo[0].pSignalSemaphores = NULL;

    //Queue the command buffer for execution
    res = vkQueueSubmit(queue, 1, submitInfo, drawFence);
    if (res != VK_SUCCESS) {
      printf ("vkQueueSubmit returned error %d.\n", res);
      return -1;
    }  

    //This waits for the queue to finish (this also involves waiting for vsync as the first buffer in this queue will wait on on the presntcomplete semaphore to be singled by vkAcquireNextImageKHR before starting).
    int timeoutCount = 0;
    do {      
	res = vkWaitForFences(device, 1, &drawFence, VK_TRUE, 100000000);
	timeoutCount++;
    } while (res == VK_TIMEOUT && timeoutCount < 10);
    
    vkDestroyFence(device, drawFence, NULL);
    /*
    //Simpler approach to using a fence is to use vkQueueWaitIdle (does not work correctly on Nvidia).
    res = vkQueueWaitIdle(queue);
    if (res != VK_SUCCESS) {
      printf ("vkQueueSubmit returned error %d.\n", res);
      return -1;
    }
    */
    //The queue is idle, now is a good time to update the bound memory.
    rotate_matrix(45+frame, 0,1,0, modelMatrix);
    multiply_matrix(viewMatrix, modelMatrix, MVMatrix);
    //As the memory is still mapped we can write the result stright into uniformMappedMemory:
    multiply_matrix(projectionMatrix, MVMatrix, (float*)uniformMappedMemory);
    
    //printf ("Command buffer finished %d.\n", res);
    
    VkPresentInfoKHR present;
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.pNext = NULL;
    present.swapchainCount = 1;
    present.pSwapchains = &swapchain;
    present.pImageIndices = &currentBuffer;
    present.pWaitSemaphores = NULL;
    present.waitSemaphoreCount = 0;
    present.pResults = NULL;
    res = vkQueuePresentKHR(queue, &present);
    if (res != VK_SUCCESS) {
      printf ("vkQueuePresentKHR returned error %d.\n", res);
      return -1;
    }

    frame++;
  }
  return 0;
}
