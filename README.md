# Vulkan Demos

These programs demonstrate some of the key features of Vulkan. Each demo is self-contained and most only depend on Vulkan & XCB.

##Building
1 Install the Vulkan sdk

2 Install the other dependencies with:

```sudo apt-get install cmake libxcb1-dev libxcb-icccm4-dev```

3 Build all demos my running cmake and make.

Since version 1.0.13 of the LinarG Vulkan sdk the necessary files are no longer automatically installed to the system directories.
You may wish to add the folowing lines to the cmake file:

```cmake
set(VULKAN_SDK_PATH /home/yourusername/Downloads/VulkanSDK/1.0.13.0/x86_64/)
include_directories(${VULKAN_SDK_PATH}/include)
link_directories(${VULKAN_SDK_PATH}/lib)
```
