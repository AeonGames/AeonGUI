/*
Copyright (C) 2026 Rodrigo Jose Hernandez Cordoba

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef VULKAN_RENDERER_H
#define VULKAN_RENDERER_H

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "aeongui/dom/Window.hpp"

#define VK_CHECK(call) \
    do { \
        VkResult result_ = (call); \
        if (result_ != VK_SUCCESS) { \
            std::cerr << "Vulkan error " << result_ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error("Vulkan call failed"); \
        } \
    } while(0)

struct Vertex
{
    float position[2];
    float texCoord[2];
};

static const Vertex kVertices[] =
{
    { {-1.0f,  1.0f}, {0.0f, 1.0f} },
    { {-1.0f, -1.0f}, {0.0f, 0.0f} },
    { { 1.0f, -1.0f}, {1.0f, 0.0f} },
    { { 1.0f,  1.0f}, {1.0f, 1.0f} },
};

static const uint16_t kIndices[] = { 0, 1, 2, 0, 2, 3 };

// ─────────────────────────────────────────────────────────────────────────────
// Helper: read a SPIR-V file into a uint32_t vector
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<uint32_t> ReadSPIRV ( const std::string& path )
{
    std::ifstream file ( path, std::ios::ate | std::ios::binary );
    if ( !file.is_open() )
    {
        throw std::runtime_error ( "Failed to open SPIR-V file: " + path );
    }
    size_t fileSize = static_cast<size_t> ( file.tellg() );
    std::vector<uint32_t> buffer ( fileSize / sizeof ( uint32_t ) );
    file.seekg ( 0 );
    file.read ( reinterpret_cast<char*> ( buffer.data() ), fileSize );
    return buffer;
}

// ─────────────────────────────────────────────────────────────────────────────
// VulkanRenderer – self-contained Vulkan rendering context
// ─────────────────────────────────────────────────────────────────────────────
class VulkanRenderer
{
public:
    VulkanRenderer() = default;
    ~VulkanRenderer()
    {
        Cleanup();
    }

    // ── Phase 1: create instance + surface (caller supplies platform surface) ──
    void CreateInstance ( const std::vector<const char*>& extraExtensions )
    {
        VkApplicationInfo appInfo{};
        appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName   = "AeonGUI – Vulkan";
        appInfo.applicationVersion = VK_MAKE_VERSION ( 1, 0, 0 );
        appInfo.pEngineName        = "AeonGUI";
        appInfo.engineVersion      = VK_MAKE_VERSION ( 1, 0, 0 );
        appInfo.apiVersion         = VK_API_VERSION_1_0;

        std::vector<const char*> extensions = extraExtensions;
        extensions.push_back ( VK_KHR_SURFACE_EXTENSION_NAME );
#ifdef __APPLE__
        extensions.push_back ( VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME );
        extensions.push_back ( "VK_KHR_get_physical_device_properties2" );
#endif

        VkInstanceCreateInfo createInfo{};
        createInfo.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo        = &appInfo;
        createInfo.enabledExtensionCount   = static_cast<uint32_t> ( extensions.size() );
        createInfo.ppEnabledExtensionNames = extensions.data();
#ifdef __APPLE__
        createInfo.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

        VK_CHECK ( vkCreateInstance ( &createInfo, nullptr, &mInstance ) );
    }

    // Surface is created externally (platform-specific) and handed in.
    void SetSurface ( VkSurfaceKHR surface )
    {
        mSurface = surface;
    }

    // ── Phase 2: pick device, create logical device, command pool ──────────
    void CreateDevice()
    {
        // Pick physical device
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices ( mInstance, &deviceCount, nullptr );
        if ( deviceCount == 0 )
        {
            throw std::runtime_error ( "No Vulkan devices found" );
        }
        std::vector<VkPhysicalDevice> devices ( deviceCount );
        vkEnumeratePhysicalDevices ( mInstance, &deviceCount, devices.data() );
        mPhysicalDevice = devices[0]; // pick first

        // Find queue families
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties ( mPhysicalDevice, &queueFamilyCount, nullptr );
        std::vector<VkQueueFamilyProperties> queueFamilies ( queueFamilyCount );
        vkGetPhysicalDeviceQueueFamilyProperties ( mPhysicalDevice, &queueFamilyCount, queueFamilies.data() );

        mGraphicsFamily = UINT32_MAX;
        mPresentFamily  = UINT32_MAX;
        for ( uint32_t i = 0; i < queueFamilyCount; i++ )
        {
            if ( queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT )
            {
                mGraphicsFamily = i;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR ( mPhysicalDevice, i, mSurface, &presentSupport );
            if ( presentSupport )
            {
                mPresentFamily = i;
            }
            if ( mGraphicsFamily != UINT32_MAX && mPresentFamily != UINT32_MAX )
            {
                break;
            }
        }

        // Logical device
        float queuePriority = 1.0f;
        std::vector<VkDeviceQueueCreateInfo> queueInfos;
        std::vector<uint32_t> uniqueFamilies = { mGraphicsFamily };
        if ( mPresentFamily != mGraphicsFamily )
        {
            uniqueFamilies.push_back ( mPresentFamily );
        }

        for ( uint32_t fam : uniqueFamilies )
        {
            VkDeviceQueueCreateInfo qi{};
            qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qi.queueFamilyIndex = fam;
            qi.queueCount       = 1;
            qi.pQueuePriorities = &queuePriority;
            queueInfos.push_back ( qi );
        }

        std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef __APPLE__
        deviceExtensions.push_back ( "VK_KHR_portability_subset" );
#endif
        VkPhysicalDeviceFeatures deviceFeatures {};

        VkDeviceCreateInfo devInfo{};
        devInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        devInfo.queueCreateInfoCount    = static_cast<uint32_t> ( queueInfos.size() );
        devInfo.pQueueCreateInfos       = queueInfos.data();
        devInfo.pEnabledFeatures        = &deviceFeatures;
        devInfo.enabledExtensionCount   = static_cast<uint32_t> ( deviceExtensions.size() );
        devInfo.ppEnabledExtensionNames = deviceExtensions.data();
        VK_CHECK ( vkCreateDevice ( mPhysicalDevice, &devInfo, nullptr, &mDevice ) );

        vkGetDeviceQueue ( mDevice, mGraphicsFamily, 0, &mGraphicsQueue );
        vkGetDeviceQueue ( mDevice, mPresentFamily,  0, &mPresentQueue );

        // Command pool
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = mGraphicsFamily;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK ( vkCreateCommandPool ( mDevice, &poolInfo, nullptr, &mCommandPool ) );
    }

    // ── Phase 3: swapchain, renderpass, pipeline, buffers, texture ─────────
    void CreateSwapchain ( uint32_t width, uint32_t height )
    {
        VkSurfaceCapabilitiesKHR caps;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR ( mPhysicalDevice, mSurface, &caps );

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR ( mPhysicalDevice, mSurface, &formatCount, nullptr );
        std::vector<VkSurfaceFormatKHR> formats ( formatCount );
        vkGetPhysicalDeviceSurfaceFormatsKHR ( mPhysicalDevice, mSurface, &formatCount, formats.data() );

        // Choose format
        mSwapchainFormat = formats[0];
        for ( auto& f : formats )
        {
            if ( f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR )
            {
                mSwapchainFormat = f;
                break;
            }
        }

        VkExtent2D extent = { width, height };
        if ( caps.currentExtent.width != UINT32_MAX )
        {
            extent = caps.currentExtent;
        }

        uint32_t imageCount = caps.minImageCount + 1;
        if ( caps.maxImageCount > 0 && imageCount > caps.maxImageCount )
        {
            imageCount = caps.maxImageCount;
        }

        VkSwapchainCreateInfoKHR scInfo{};
        scInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        scInfo.surface          = mSurface;
        scInfo.minImageCount    = imageCount;
        scInfo.imageFormat      = mSwapchainFormat.format;
        scInfo.imageColorSpace  = mSwapchainFormat.colorSpace;
        scInfo.imageExtent      = extent;
        scInfo.imageArrayLayers = 1;
        scInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        scInfo.preTransform     = caps.currentTransform;
        scInfo.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        scInfo.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
        scInfo.clipped          = VK_TRUE;

        uint32_t families[] = { mGraphicsFamily, mPresentFamily };
        if ( mGraphicsFamily != mPresentFamily )
        {
            scInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
            scInfo.queueFamilyIndexCount = 2;
            scInfo.pQueueFamilyIndices   = families;
        }
        else
        {
            scInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VK_CHECK ( vkCreateSwapchainKHR ( mDevice, &scInfo, nullptr, &mSwapchain ) );
        mSwapchainExtent = extent;

        vkGetSwapchainImagesKHR ( mDevice, mSwapchain, &imageCount, nullptr );
        mSwapchainImages.resize ( imageCount );
        vkGetSwapchainImagesKHR ( mDevice, mSwapchain, &imageCount, mSwapchainImages.data() );

        // Image views
        mSwapchainImageViews.resize ( mSwapchainImages.size() );
        for ( size_t i = 0; i < mSwapchainImages.size(); i++ )
        {
            VkImageViewCreateInfo ivInfo{};
            ivInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            ivInfo.image    = mSwapchainImages[i];
            ivInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivInfo.format   = mSwapchainFormat.format;
            ivInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            ivInfo.subresourceRange.baseMipLevel   = 0;
            ivInfo.subresourceRange.levelCount     = 1;
            ivInfo.subresourceRange.baseArrayLayer = 0;
            ivInfo.subresourceRange.layerCount     = 1;
            VK_CHECK ( vkCreateImageView ( mDevice, &ivInfo, nullptr, &mSwapchainImageViews[i] ) );
        }
    }

    void CreateRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format         = mSwapchainFormat.format;
        colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments    = &colorRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass    = 0;
        dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpInfo{};
        rpInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpInfo.attachmentCount = 1;
        rpInfo.pAttachments    = &colorAttachment;
        rpInfo.subpassCount    = 1;
        rpInfo.pSubpasses      = &subpass;
        rpInfo.dependencyCount = 1;
        rpInfo.pDependencies   = &dependency;
        VK_CHECK ( vkCreateRenderPass ( mDevice, &rpInfo, nullptr, &mRenderPass ) );
    }

    void CreateFramebuffers()
    {
        mFramebuffers.resize ( mSwapchainImageViews.size() );
        for ( size_t i = 0; i < mSwapchainImageViews.size(); i++ )
        {
            VkFramebufferCreateInfo fbInfo{};
            fbInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fbInfo.renderPass      = mRenderPass;
            fbInfo.attachmentCount = 1;
            fbInfo.pAttachments    = &mSwapchainImageViews[i];
            fbInfo.width           = mSwapchainExtent.width;
            fbInfo.height          = mSwapchainExtent.height;
            fbInfo.layers          = 1;
            VK_CHECK ( vkCreateFramebuffer ( mDevice, &fbInfo, nullptr, &mFramebuffers[i] ) );
        }
    }

    void CreatePipeline ( const std::string& vertSpvPath, const std::string& fragSpvPath )
    {
        auto vertCode = ReadSPIRV ( vertSpvPath );
        auto fragCode = ReadSPIRV ( fragSpvPath );

        VkShaderModule vertModule = CreateShaderModule ( vertCode );
        VkShaderModule fragModule = CreateShaderModule ( fragCode );

        VkPipelineShaderStageCreateInfo stages[2] {};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertModule;
        stages[0].pName  = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragModule;
        stages[1].pName  = "main";

        // Vertex input
        VkVertexInputBindingDescription bindingDesc{};
        bindingDesc.binding   = 0;
        bindingDesc.stride    = sizeof ( Vertex );
        bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrDesc[2] {};
        attrDesc[0].binding  = 0;
        attrDesc[0].location = 0;
        attrDesc[0].format   = VK_FORMAT_R32G32_SFLOAT;
        attrDesc[0].offset   = offsetof ( Vertex, position );
        attrDesc[1].binding  = 0;
        attrDesc[1].location = 1;
        attrDesc[1].format   = VK_FORMAT_R32G32_SFLOAT;
        attrDesc[1].offset   = offsetof ( Vertex, texCoord );

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount   = 1;
        vertexInput.pVertexBindingDescriptions      = &bindingDesc;
        vertexInput.vertexAttributeDescriptionCount = 2;
        vertexInput.pVertexAttributeDescriptions    = attrDesc;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.width    = static_cast<float> ( mSwapchainExtent.width );
        viewport.height   = static_cast<float> ( mSwapchainExtent.height );
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.extent = mSwapchainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports    = &viewport;
        viewportState.scissorCount  = 1;
        viewportState.pScissors     = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth   = 1.0f;
        rasterizer.cullMode    = VK_CULL_MODE_NONE;
        rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.blendEnable         = VK_TRUE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;
        blendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &blendAttachment;

        // Descriptor set layout (one combined image sampler)
        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding         = 0;
        samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount = 1;
        samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dsLayoutInfo{};
        dsLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dsLayoutInfo.bindingCount = 1;
        dsLayoutInfo.pBindings    = &samplerBinding;
        VK_CHECK ( vkCreateDescriptorSetLayout ( mDevice, &dsLayoutInfo, nullptr, &mDescriptorSetLayout ) );

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts    = &mDescriptorSetLayout;
        VK_CHECK ( vkCreatePipelineLayout ( mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout ) );

        VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates    = dynamicStates;

        VkGraphicsPipelineCreateInfo pipeInfo{};
        pipeInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeInfo.stageCount          = 2;
        pipeInfo.pStages             = stages;
        pipeInfo.pVertexInputState   = &vertexInput;
        pipeInfo.pInputAssemblyState = &inputAssembly;
        pipeInfo.pViewportState      = &viewportState;
        pipeInfo.pRasterizationState = &rasterizer;
        pipeInfo.pMultisampleState   = &multisampling;
        pipeInfo.pColorBlendState    = &colorBlending;
        pipeInfo.pDynamicState       = &dynamicState;
        pipeInfo.layout              = mPipelineLayout;
        pipeInfo.renderPass          = mRenderPass;
        pipeInfo.subpass             = 0;
        VK_CHECK ( vkCreateGraphicsPipelines ( mDevice, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &mPipeline ) );

        vkDestroyShaderModule ( mDevice, vertModule, nullptr );
        vkDestroyShaderModule ( mDevice, fragModule, nullptr );
    }

    void CreateVertexAndIndexBuffers()
    {
        mVertexBuffer = CreateBufferWithData ( VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                               kVertices, sizeof ( kVertices ) );
        mIndexBuffer  = CreateBufferWithData ( VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                               kIndices, sizeof ( kIndices ) );
    }

    void CreateTextureResources ( uint32_t texWidth, uint32_t texHeight )
    {
        mTexWidth  = texWidth;
        mTexHeight = texHeight;

        // Staging buffer (host-visible, for CPU uploads each frame)
        VkDeviceSize imageSize = static_cast<VkDeviceSize> ( texWidth ) * texHeight * 4;
        CreateBuffer ( imageSize,
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       mStagingBuffer, mStagingBufferMemory );
        vkMapMemory ( mDevice, mStagingBufferMemory, 0, imageSize, 0, &mStagingMapped );

        // Image
        VkImageCreateInfo imgInfo{};
        imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imgInfo.imageType     = VK_IMAGE_TYPE_2D;
        imgInfo.format        = VK_FORMAT_B8G8R8A8_UNORM;
        imgInfo.extent        = { texWidth, texHeight, 1 };
        imgInfo.mipLevels     = 1;
        imgInfo.arrayLayers   = 1;
        imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        imgInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK ( vkCreateImage ( mDevice, &imgInfo, nullptr, &mTextureImage ) );

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements ( mDevice, mTextureImage, &memReqs );
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReqs.size;
        allocInfo.memoryTypeIndex = FindMemoryType ( memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT );
        VK_CHECK ( vkAllocateMemory ( mDevice, &allocInfo, nullptr, &mTextureImageMemory ) );
        vkBindImageMemory ( mDevice, mTextureImage, mTextureImageMemory, 0 );

        // Image view
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = mTextureImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = VK_FORMAT_B8G8R8A8_UNORM;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;
        VK_CHECK ( vkCreateImageView ( mDevice, &viewInfo, nullptr, &mTextureImageView ) );

        // Sampler
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType     = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK ( vkCreateSampler ( mDevice, &samplerInfo, nullptr, &mTextureSampler ) );
    }

    void CreateDescriptorSets()
    {
        // Pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 1;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes    = &poolSize;
        poolInfo.maxSets       = 1;
        VK_CHECK ( vkCreateDescriptorPool ( mDevice, &poolInfo, nullptr, &mDescriptorPool ) );

        // Allocate
        VkDescriptorSetAllocateInfo dsAlloc{};
        dsAlloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsAlloc.descriptorPool     = mDescriptorPool;
        dsAlloc.descriptorSetCount = 1;
        dsAlloc.pSetLayouts        = &mDescriptorSetLayout;
        VK_CHECK ( vkAllocateDescriptorSets ( mDevice, &dsAlloc, &mDescriptorSet ) );

        // Write
        VkDescriptorImageInfo imgDescInfo{};
        imgDescInfo.sampler     = mTextureSampler;
        imgDescInfo.imageView   = mTextureImageView;
        imgDescInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = mDescriptorSet;
        write.dstBinding      = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &imgDescInfo;
        vkUpdateDescriptorSets ( mDevice, 1, &write, 0, nullptr );
    }

    void CreateCommandBufferAndSync()
    {
        VkCommandBufferAllocateInfo cbAlloc{};
        cbAlloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbAlloc.commandPool        = mCommandPool;
        cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbAlloc.commandBufferCount = 1;
        VK_CHECK ( vkAllocateCommandBuffers ( mDevice, &cbAlloc, &mCommandBuffer ) );

        VkSemaphoreCreateInfo semInfo{};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK ( vkCreateSemaphore ( mDevice, &semInfo, nullptr, &mImageAvailable ) );
        VK_CHECK ( vkCreateSemaphore ( mDevice, &semInfo, nullptr, &mRenderFinished ) );

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK ( vkCreateFence ( mDevice, &fenceInfo, nullptr, &mInFlightFence ) );
    }

    // ── Full init helper (call after surface creation) ─────────────────────
    void Initialize ( uint32_t width, uint32_t height,
                      const std::string& vertSpvPath,
                      const std::string& fragSpvPath,
                      uint32_t texWidth, uint32_t texHeight )
    {
        CreateDevice();
        CreateSwapchain ( width, height );
        CreateRenderPass();
        CreateFramebuffers();
        CreatePipeline ( vertSpvPath, fragSpvPath );
        CreateVertexAndIndexBuffers();
        CreateTextureResources ( texWidth, texHeight );
        CreateDescriptorSets();
        CreateCommandBufferAndSync();
    }

    // ── Draw one frame ─────────────────────────────────────────────────────
    void DrawFrame ( AeonGUI::DOM::Window& window )
    {
        vkWaitForFences ( mDevice, 1, &mInFlightFence, VK_TRUE, UINT64_MAX );
        vkResetFences ( mDevice, 1, &mInFlightFence );

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR ( mDevice, mSwapchain, UINT64_MAX,
                          mImageAvailable, VK_NULL_HANDLE, &imageIndex );
        if ( result == VK_ERROR_OUT_OF_DATE_KHR )
        {
            // Swapchain needs recreation – caller should handle resize
            return;
        }

        window.Draw();

        // Upload pixels into staging buffer, then copy to image
        UploadTexture ( window );

        // Record command buffer
        vkResetCommandBuffer ( mCommandBuffer, 0 );
        RecordCommandBuffer ( imageIndex );

        // Submit
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount   = 1;
        submitInfo.pWaitSemaphores      = &mImageAvailable;
        submitInfo.pWaitDstStageMask    = &waitStage;
        submitInfo.commandBufferCount   = 1;
        submitInfo.pCommandBuffers      = &mCommandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores    = &mRenderFinished;
        VK_CHECK ( vkQueueSubmit ( mGraphicsQueue, 1, &submitInfo, mInFlightFence ) );

        // Present
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores    = &mRenderFinished;
        presentInfo.swapchainCount     = 1;
        presentInfo.pSwapchains        = &mSwapchain;
        presentInfo.pImageIndices      = &imageIndex;
        vkQueuePresentKHR ( mPresentQueue, &presentInfo );

        // Ensure present completes before next frame reuses mRenderFinished semaphore
        vkQueueWaitIdle ( mPresentQueue );
    }

    // ── Resize handling ────────────────────────────────────────────────────
    void RecreateSwapchain ( uint32_t width, uint32_t height,
                             AeonGUI::DOM::Window& window,
                             const std::string& vertSpvPath,
                             const std::string& fragSpvPath )
    {
        vkDeviceWaitIdle ( mDevice );
        CleanupSwapchain();
        CleanupTextureResources();
        CleanupPipeline();

        window.ResizeViewport ( width, height );

        CreateSwapchain ( width, height );
        CreateRenderPass();
        CreateFramebuffers();
        CreatePipeline ( vertSpvPath, fragSpvPath );
        CreateTextureResources (
            static_cast<uint32_t> ( window.GetWidth() ),
            static_cast<uint32_t> ( window.GetHeight() ) );
        CreateDescriptorSets();
    }

    VkInstance GetInstance() const
    {
        return mInstance;
    }
    VkDevice   GetDevice()  const
    {
        return mDevice;
    }

    void Cleanup()
    {
        if ( mDevice )
        {
            vkDeviceWaitIdle ( mDevice );
        }

        if ( mInFlightFence )
        {
            vkDestroyFence ( mDevice, mInFlightFence, nullptr );
            mInFlightFence   = VK_NULL_HANDLE;
        }
        if ( mRenderFinished )
        {
            vkDestroySemaphore ( mDevice, mRenderFinished, nullptr );
            mRenderFinished  = VK_NULL_HANDLE;
        }
        if ( mImageAvailable )
        {
            vkDestroySemaphore ( mDevice, mImageAvailable, nullptr );
            mImageAvailable  = VK_NULL_HANDLE;
        }
        if ( mCommandPool )
        {
            vkDestroyCommandPool ( mDevice, mCommandPool, nullptr );
            mCommandPool     = VK_NULL_HANDLE;
        }

        CleanupPipeline();
        CleanupTextureResources();
        CleanupSwapchain();

        if ( mDevice )
        {
            vkDestroyDevice ( mDevice, nullptr );
            mDevice   = VK_NULL_HANDLE;
        }
        if ( mSurface )
        {
            vkDestroySurfaceKHR ( mInstance, mSurface, nullptr );
            mSurface  = VK_NULL_HANDLE;
        }
        if ( mInstance )
        {
            vkDestroyInstance ( mInstance, nullptr );
            mInstance = VK_NULL_HANDLE;
        }
    }

private:
    VkShaderModule CreateShaderModule ( const std::vector<uint32_t>& code )
    {
        VkShaderModuleCreateInfo info{};
        info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = code.size() * sizeof ( uint32_t );
        info.pCode    = code.data();
        VkShaderModule module;
        VK_CHECK ( vkCreateShaderModule ( mDevice, &info, nullptr, &module ) );
        return module;
    }

    uint32_t FindMemoryType ( uint32_t typeFilter, VkMemoryPropertyFlags properties )
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties ( mPhysicalDevice, &memProps );
        for ( uint32_t i = 0; i < memProps.memoryTypeCount; i++ )
        {
            if ( ( typeFilter & ( 1 << i ) ) &&
                 ( memProps.memoryTypes[i].propertyFlags & properties ) == properties )
            {
                return i;
            }
        }
        throw std::runtime_error ( "Failed to find suitable memory type" );
    }

    void CreateBuffer ( VkDeviceSize size, VkBufferUsageFlags usage,
                        VkMemoryPropertyFlags properties,
                        VkBuffer& buffer, VkDeviceMemory& memory )
    {
        VkBufferCreateInfo bufInfo{};
        bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufInfo.size  = size;
        bufInfo.usage = usage;
        bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK ( vkCreateBuffer ( mDevice, &bufInfo, nullptr, &buffer ) );

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements ( mDevice, buffer, &memReqs );
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReqs.size;
        allocInfo.memoryTypeIndex = FindMemoryType ( memReqs.memoryTypeBits, properties );
        VK_CHECK ( vkAllocateMemory ( mDevice, &allocInfo, nullptr, &memory ) );
        vkBindBufferMemory ( mDevice, buffer, memory, 0 );
    }

    VkBuffer CreateBufferWithData ( VkBufferUsageFlags usage, const void* data, VkDeviceSize size )
    {
        VkBuffer buffer;
        VkDeviceMemory memory;
        CreateBuffer ( size, usage,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                       buffer, memory );
        void* mapped;
        vkMapMemory ( mDevice, memory, 0, size, 0, &mapped );
        memcpy ( mapped, data, size );
        vkUnmapMemory ( mDevice, memory );
        // NOTE: we leak device memory here for simplicity; a proper implementation
        // would track and free it. For this small demo it is acceptable.
        if ( usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT )
        {
            mVertexBufferMemory = memory;
        }
        else
        {
            mIndexBufferMemory = memory;
        }
        return buffer;
    }

    void UploadTexture ( AeonGUI::DOM::Window& window )
    {
        size_t dataSize = window.GetHeight() * window.GetStride();
        memcpy ( mStagingMapped, window.GetPixels(), dataSize );
    }

    void RecordCommandBuffer ( uint32_t imageIndex )
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        VK_CHECK ( vkBeginCommandBuffer ( mCommandBuffer, &beginInfo ) );

        // Transition texture image to TRANSFER_DST
        TransitionImageLayout ( mCommandBuffer, mTextureImage,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL );

        // Copy staging buffer → texture image
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = { mTexWidth, mTexHeight, 1 };
        vkCmdCopyBufferToImage ( mCommandBuffer, mStagingBuffer, mTextureImage,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region );

        // Transition texture image to SHADER_READ_ONLY
        TransitionImageLayout ( mCommandBuffer, mTextureImage,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        // Begin render pass
        VkClearValue clearColor = {{{1.0f, 1.0f, 1.0f, 1.0f}}};
        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass        = mRenderPass;
        rpBegin.framebuffer       = mFramebuffers[imageIndex];
        rpBegin.renderArea.extent = mSwapchainExtent;
        rpBegin.clearValueCount   = 1;
        rpBegin.pClearValues      = &clearColor;
        vkCmdBeginRenderPass ( mCommandBuffer, &rpBegin, VK_SUBPASS_CONTENTS_INLINE );

        vkCmdBindPipeline ( mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mPipeline );

        VkViewport viewport{};
        viewport.width    = static_cast<float> ( mSwapchainExtent.width );
        viewport.height   = static_cast<float> ( mSwapchainExtent.height );
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport ( mCommandBuffer, 0, 1, &viewport );

        VkRect2D scissor{};
        scissor.extent = mSwapchainExtent;
        vkCmdSetScissor ( mCommandBuffer, 0, 1, &scissor );

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers ( mCommandBuffer, 0, 1, &mVertexBuffer, offsets );
        vkCmdBindIndexBuffer ( mCommandBuffer, mIndexBuffer, 0, VK_INDEX_TYPE_UINT16 );
        vkCmdBindDescriptorSets ( mCommandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  mPipelineLayout, 0, 1, &mDescriptorSet, 0, nullptr );
        vkCmdDrawIndexed ( mCommandBuffer, 6, 1, 0, 0, 0 );

        vkCmdEndRenderPass ( mCommandBuffer );
        VK_CHECK ( vkEndCommandBuffer ( mCommandBuffer ) );
    }

    static void TransitionImageLayout ( VkCommandBuffer cmd, VkImage image,
                                        VkImageLayout oldLayout, VkImageLayout newLayout )
    {
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = oldLayout;
        barrier.newLayout           = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = image;
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;

        VkPipelineStageFlags srcStage, dstStage;

        if ( oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
             newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL )
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if ( oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                  newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL )
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            srcStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }

        vkCmdPipelineBarrier ( cmd, srcStage, dstStage, 0,
                               0, nullptr, 0, nullptr, 1, &barrier );
    }

    void CleanupSwapchain()
    {
        for ( auto fb : mFramebuffers )
        {
            vkDestroyFramebuffer ( mDevice, fb, nullptr );
        }
        mFramebuffers.clear();
        for ( auto iv : mSwapchainImageViews )
        {
            vkDestroyImageView ( mDevice, iv, nullptr );
        }
        mSwapchainImageViews.clear();
        if ( mRenderPass )
        {
            vkDestroyRenderPass ( mDevice, mRenderPass, nullptr );
            mRenderPass = VK_NULL_HANDLE;
        }
        if ( mSwapchain )
        {
            vkDestroySwapchainKHR ( mDevice, mSwapchain, nullptr );
            mSwapchain = VK_NULL_HANDLE;
        }
    }

    void CleanupTextureResources()
    {
        if ( mDescriptorPool )
        {
            vkDestroyDescriptorPool ( mDevice, mDescriptorPool, nullptr );
            mDescriptorPool    = VK_NULL_HANDLE;
        }
        if ( mTextureSampler )
        {
            vkDestroySampler ( mDevice, mTextureSampler, nullptr );
            mTextureSampler    = VK_NULL_HANDLE;
        }
        if ( mTextureImageView )
        {
            vkDestroyImageView ( mDevice, mTextureImageView, nullptr );
            mTextureImageView  = VK_NULL_HANDLE;
        }
        if ( mTextureImage )
        {
            vkDestroyImage ( mDevice, mTextureImage, nullptr );
            mTextureImage      = VK_NULL_HANDLE;
        }
        if ( mTextureImageMemory )
        {
            vkFreeMemory ( mDevice, mTextureImageMemory, nullptr );
            mTextureImageMemory = VK_NULL_HANDLE;
        }
        if ( mStagingBuffer )
        {
            vkDestroyBuffer ( mDevice, mStagingBuffer, nullptr );
            mStagingBuffer     = VK_NULL_HANDLE;
        }
        if ( mStagingBufferMemory )
        {
            vkFreeMemory ( mDevice, mStagingBufferMemory, nullptr );
            mStagingBufferMemory = VK_NULL_HANDLE;
        }
        mStagingMapped = nullptr;
    }

    void CleanupPipeline()
    {
        if ( mPipeline )
        {
            vkDestroyPipeline ( mDevice, mPipeline, nullptr );
            mPipeline = VK_NULL_HANDLE;
        }
        if ( mPipelineLayout )
        {
            vkDestroyPipelineLayout ( mDevice, mPipelineLayout, nullptr );
            mPipelineLayout = VK_NULL_HANDLE;
        }
        if ( mDescriptorSetLayout )
        {
            vkDestroyDescriptorSetLayout ( mDevice, mDescriptorSetLayout, nullptr );
            mDescriptorSetLayout = VK_NULL_HANDLE;
        }
    }

    // ── Members ──────────────────────────────────────────────────────────────
    VkInstance               mInstance            = VK_NULL_HANDLE;
    VkSurfaceKHR             mSurface             = VK_NULL_HANDLE;
    VkPhysicalDevice         mPhysicalDevice      = VK_NULL_HANDLE;
    VkDevice                 mDevice              = VK_NULL_HANDLE;
    VkQueue                  mGraphicsQueue       = VK_NULL_HANDLE;
    VkQueue                  mPresentQueue        = VK_NULL_HANDLE;
    uint32_t                 mGraphicsFamily      = UINT32_MAX;
    uint32_t                 mPresentFamily       = UINT32_MAX;
    VkCommandPool            mCommandPool         = VK_NULL_HANDLE;

    VkSwapchainKHR           mSwapchain           = VK_NULL_HANDLE;
    VkSurfaceFormatKHR       mSwapchainFormat     = {};
    VkExtent2D               mSwapchainExtent     = {};
    std::vector<VkImage>     mSwapchainImages;
    std::vector<VkImageView> mSwapchainImageViews;
    std::vector<VkFramebuffer> mFramebuffers;

    VkRenderPass             mRenderPass          = VK_NULL_HANDLE;
    VkDescriptorSetLayout    mDescriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout         mPipelineLayout      = VK_NULL_HANDLE;
    VkPipeline               mPipeline            = VK_NULL_HANDLE;

    VkBuffer                 mVertexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory           mVertexBufferMemory  = VK_NULL_HANDLE;
    VkBuffer                 mIndexBuffer         = VK_NULL_HANDLE;
    VkDeviceMemory           mIndexBufferMemory   = VK_NULL_HANDLE;

    VkBuffer                 mStagingBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory           mStagingBufferMemory = VK_NULL_HANDLE;
    void*                    mStagingMapped       = nullptr;
    VkImage                  mTextureImage        = VK_NULL_HANDLE;
    VkDeviceMemory           mTextureImageMemory  = VK_NULL_HANDLE;
    VkImageView              mTextureImageView    = VK_NULL_HANDLE;
    VkSampler                mTextureSampler      = VK_NULL_HANDLE;
    uint32_t                 mTexWidth             = 0;
    uint32_t                 mTexHeight            = 0;

    VkDescriptorPool         mDescriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet          mDescriptorSet       = VK_NULL_HANDLE;

    VkCommandBuffer          mCommandBuffer       = VK_NULL_HANDLE;
    VkSemaphore              mImageAvailable      = VK_NULL_HANDLE;
    VkSemaphore              mRenderFinished      = VK_NULL_HANDLE;
    VkFence                  mInFlightFence        = VK_NULL_HANDLE;
};

#endif // VULKAN_RENDERER_H
