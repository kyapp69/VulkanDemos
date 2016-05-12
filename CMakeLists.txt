cmake_minimum_required (VERSION 2.6)
project (VulkanDemos)

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_C_FLAGS "--std=gnu99 ${CMAKE_C_FLAGS}")
endif ()

add_executable(vulkanCombined vulkanCombined.c)
add_executable(vulkanCoords vulkanCoords.c)
add_executable(vulkanResizable vulkanResizable.c)
add_executable(vulkanReuseCommandBuffer vulkanReuseCommandBuffer.c)
add_executable(vulkanUniforms vulkanUniforms.c)
add_executable(vulkanIns vulkanIns.c)
add_executable(vulkanTexture vulkanTexture.c)

target_link_libraries(vulkanCombined vulkan xcb xcb-icccm m)
target_link_libraries(vulkanCoords vulkan xcb xcb-icccm m)
target_link_libraries(vulkanResizable vulkan xcb m)
target_link_libraries(vulkanReuseCommandBuffer vulkan xcb xcb-icccm m)
target_link_libraries(vulkanUniforms vulkan xcb m)
target_link_libraries(vulkanIns vulkan xcb m)
target_link_libraries(vulkanTexture vulkan xcb m)

