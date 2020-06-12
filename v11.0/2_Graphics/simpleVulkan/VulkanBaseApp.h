/*
 * Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__

#include <string>
#include <vector>
#include <vulkan/vulkan.h>
#ifdef _WIN64
#define NOMINMAX
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif /* _WIN64 */

struct GLFWwindow;

class VulkanBaseApp
{
public:
    VulkanBaseApp(const std::string& appName, bool enableValidation = false);
    static VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType();
    static VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType();
    virtual ~VulkanBaseApp();
    void init();
    void *getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType);
    void *getSemaphoreHandle(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void importExternalBuffer(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory);
    void copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void mainLoop();
protected:
    const std::string m_appName;
    const bool m_enableValidation;
    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    VkSurfaceKHR m_surface;
    VkPhysicalDevice m_physicalDevice;
    VkDevice m_device;
    VkQueue m_graphicsQueue;
    VkQueue m_presentQueue;
    VkSwapchainKHR m_swapChain;
    std::vector<VkImage> m_swapChainImages;
    VkFormat m_swapChainFormat;
    VkExtent2D m_swapChainExtent;
    std::vector<VkImageView> m_swapChainImageViews;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > m_shaderFiles;
    VkRenderPass m_renderPass;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;
    std::vector<VkFramebuffer> m_swapChainFramebuffers;
    VkCommandPool m_commandPool;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    std::vector<VkBuffer> m_uniformBuffers;
    std::vector<VkDeviceMemory> m_uniformMemory;
    VkDescriptorSetLayout m_descriptorSetLayout;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_descriptorSets;
    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;
    size_t m_currentFrame;
    bool m_framebufferResized;
    uint8_t  m_vkDeviceUUID[VK_UUID_SIZE];

    virtual void initVulkanApp() {}
    virtual void fillRenderingCommandBuffer(VkCommandBuffer& buffer) {}
    virtual std::vector<const char *> getRequiredExtensions() const;
    virtual std::vector<const char *> getRequiredDeviceExtensions() const;
    virtual void getVertexDescriptions(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
    virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const;
    virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
    virtual VkDeviceSize getUniformSize() const;
    virtual void updateUniformBuffer(uint32_t imageIndex);
    virtual void drawFrame();
private:
    GLFWwindow *m_window;

    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

    void cleanupSwapChain();
    void recreateSwapChain();

    bool isSuitableDevice(VkPhysicalDevice dev) const;
    static void resizeCallback(GLFWwindow *window, int width, int height);
};

void readFile(std::istream& s, std::vector<char>& data);

#endif /* __VULKANBASEAPP_H__ */
