#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
//#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <optional>
#include <set>
#include <cstdint>
#include <algorithm>
#include <fstream>

/**
 * Misc notes:
 *
 */

const int WIDTH = 800;
const int HEIGHT = 600;
const char * TITLE = "Hello Triangle";
const std::vector<const char*> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

static std::vector<char> readFile(const std::string& filename)
{
    // ate: Start reading at the end of the file
    // binary: Read the file as binary file (avoid text transformations)
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("failed to open file!");
    }

    // allocate buffer from file size
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    // seek to beginning and read all bytes
    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();

    return buffer;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr) {
        func(instance, callback, pAllocator);
    }
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication
{
    public:
        void run()
        {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }

    private:
        GLFWwindow * window;
        VkInstance instance;
        VkDebugUtilsMessengerEXT callback;

        VkSurfaceKHR surface;

        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device;            // logical device

        VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkSwapchainKHR swapChain;
        std::vector<VkImage> swapChainImages;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;

        std::vector<VkImageView> swapChainImageViews;

        VkRenderPass renderPass;
        VkPipelineLayout pipelineLayout;

        VkPipeline graphicsPipeline;

        std::vector<VkFramebuffer> swapChainFramebuffers;

        void initWindow()
        {
            glfwInit();
            // tell glfw to not use opengl
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            // disable window resizing
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            window = glfwCreateWindow(WIDTH, HEIGHT, TITLE, nullptr, nullptr);
        }

        void initVulkan()
        {
            createInstance();
            setupDebugCallback();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
            createRenderPass();
	    createGraphicsPipeline();
            createFramebuffers();
        }

        void createFramebuffers()
        {
            // resize container to hold all framebuffers
            swapChainFramebuffers.resize(swapChainImageViews.size());

            // create framebuffers from image views
            for (size_t i = 0; i < swapChainImageViews.size(); i++)
            {
                VkImageView attachments[] = {
                    swapChainImageViews[i]
                };

                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                // must be compatible with render pass - roughly means same number and type of attachments
                framebufferInfo.renderPass = renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = swapChainExtent.width;
                framebufferInfo.height = swapChainExtent.height;
                // number of layers in image arrays
                framebufferInfo.layers = 1;

                if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create framebuffer!");
                }
            }
        }


        void createRenderPass()
        {
            // single color buffer attachement represented by image from swap chain
            VkAttachmentDescription colorAttachment = {};
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            // The loadOp and storeOp determine what to do with the data in the attachment before rendering and after rendering
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            // same, but for stencil data
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            // single subpass
            // each subpass references one or more attachments
            VkAttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass = {};
            // specify that this is a graphics subpass
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            // create render pass
            // structure with array of subpasses and attachements
            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;

            if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create render pass!");
            }
        }

        /**
         * The following objects combine to define the graphics pipeline:
         * Shader stages: the shader modules that define the functionality of the programmable stages of the graphics pipeline
         * Fixed-function state: all of the structures that define the fixed-function stages of the pipeline, like input assembly, rasterizer, viewport and color blending
         * Pipeline layout: the uniform and push values referenced by the shader that can be updated at draw time
         * Render pass: the attachments referenced by the pipeline stages and their usage
         */
        void createGraphicsPipeline()
        {
	    // can destroy shader modules after graphics pipeline is created
            auto vertShaderCode = readFile("shaders/vert.spv");
            auto fragShaderCode = readFile("shaders/frag.spv");

            VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
            VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

            // assign shaders to pipeline stage
            VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            // There is an enum value for each of the programmable stage of pipeline
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";       // entry point function in spir-v
            // pSpecializationInfo attribute allows you to set constants in shader code

            // setup fragment shader
            VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            // for referencing later
            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            // no vertex data to load
            VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

            // configure input assembly
            VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            // what kind of geometry will be drawn from the vertices - triangles
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            // configure viewport (region of framebuffer to render to)
            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float) swapChainExtent.width;
            viewport.height = (float) swapChainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            // scissor rectangle that covers entire framebuffer
            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = swapChainExtent;

            // combine into viewport state
            VkPipelineViewportStateCreateInfo viewportState = {};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            // rasterizer
            // The rasterizer takes the geometry that is shaped by the vertices from the vertex shader and turns it into fragments to be colored by the fragment shader.
            VkPipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            // if true, fragments that are beyond the near and far planes are clamped to them as opposed to discarding them.
            // This is useful in some special cases like shadow maps. Using this requires enabling a GPU feature.
            rasterizer.depthClampEnable = VK_FALSE;
            // if true, then geometry never passes through the rasterizer stage. This basically disables any output to the framebuffer.
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            // how fragments are generated for geometry
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;       // fill the area of the polygon with fragments, using any other mode requires enabling GPU feature
            // thickness of lines in terms of number of fragments
            // greater than 1.0f requires enabling of wideLines GPU feature
            rasterizer.lineWidth = 1.0f;
            // face culling
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f; // Optional
            rasterizer.depthBiasClamp = 0.0f; // Optional
            rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

            // Multisampling
            // one way to perform antialiasing
            // works by combining the fragment shader results of multiple polygons that rasterize to the same pixel
            // enabling requires enabling a GPU feature - leaving disabled for now
            VkPipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f; // Optional
            multisampling.pSampleMask = nullptr; // Optional
            multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
            multisampling.alphaToOneEnable = VK_FALSE; // Optional

            /*
             * If you are using a depth and/or stencil buffer, then you also need to configure the depth and stencil tests 
             * using VkPipelineDepthStencilStateCreateInfo. We don't have one right now, so we can simply pass a nullptr 
             * instead of a pointer to such a struct. We'll get back to it in the depth buffering chapter.
             */

            // color blending
            /* After a fragment shader has returned a color, it needs to be combined with the color that is already in the framebuffer
             * This transformation is known as color blending and there are two ways to do it:
             * - Mix the old and new value to produce a final color
             * - Combine the old and new value using a bitwise operation
             * There are two types of structs to configure color blending.
             * The first struct, VkPipelineColorBlendAttachmentState contains the configuration per attached framebuffer 
             * and the second struct, VkPipelineColorBlendStateCreateInfo contains the global color blending settings. In our case we only have one framebuffer:
             */
            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
            // If blendEnable is set to VK_FALSE, then the new color from the fragment shader is passed through unmodified.
            // Otherwise, the two mixing operations are performed to compute a new color.
            // The resulting color is AND'd with the colorWriteMask to determine which channels are actually passed through.
            // this can be used for alpha blending (opacity)

            // second color blending structure (global color blending settings)
            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;
            colorBlending.blendConstants[0] = 0.0f; // Optional
            colorBlending.blendConstants[1] = 0.0f; // Optional
            colorBlending.blendConstants[2] = 0.0f; // Optional
            colorBlending.blendConstants[3] = 0.0f; // Optional

            // a bit of state can be changed dynamically with VkPipelineDynamicStateCreateInfo
            // to avoid recreating entire pipeline

            // create an empty pipeline layout
            // The structure also specifies push constants, which are another way of passing dynamic values to shaders
            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 0; // Optional
            pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
            pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
            pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create pipeline layout!");
            }

            // create the pipeline itself
            VkGraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;
            // reference fixed function stage
            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr; // Optional
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr; // Optional
            // set pipeline layout as vulkan handle, not struct pointer
            pipelineInfo.layout = pipelineLayout;

            pipelineInfo.renderPass = renderPass;
            pipelineInfo.subpass = 0;
            
            // you can create a new graphics pipeline by deriving from existing pipeline
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
            pipelineInfo.basePipelineIndex = -1; // Optional

            // create it!
            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create graphics pipeline!");
            }
            
	    // destroy shader modules
            vkDestroyShaderModule(device, fragShaderModule, nullptr);
            vkDestroyShaderModule(device, vertShaderModule, nullptr);
        }

        /**                                                                                                                                                                                  
         * Create shader module for spir-v bytecode                                                                                                                                          
         * Thin wrapper around shader bytecode                                                                                                                                              
         */
        VkShaderModule createShaderModule(const std::vector<char>& code)
        {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            // cast to uint32_t, must satify alignment requirements - std::vector allocator ensures
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

            // create shader module                                                 
            VkShaderModule shaderModule;
            // params: the logical device, pointer to create info structure, optional pointer to custom allocators and handle output variable
            if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create shader module!");
            }

            return shaderModule;
        }

        /**
         * create image views for every image in swap chain
         */
        void createImageViews()
        {
            swapChainImageViews.resize(swapChainImages.size());

            for(size_t i = 0; i < swapChainImages.size(); i++)
            {
                VkImageViewCreateInfo createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image = swapChainImages[i];

                // how image data should be interpretted
                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = swapChainImageFormat;

                // components field allows you to swizzle color channels, keep default
                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

                // describes what purpose of image and what part to access
                // use as color targets with no mipmapping or layering
                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                // create image view
                if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
                {
                    throw std::runtime_error("failed to create image views!");
                }
            }
        }

        /**
         * swap chain terms:
         * Surface format (color depth)
         * Presentation mode (conditions for "swapping" images to the screen)
         * Swap extent (resolution of images in swap chain)
         */
        void createSwapChain()
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

            VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
            VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
            VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

            // number of images to have in swap chain
            uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

            // ensure max number of images is not exceeded
            if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            {
                imageCount = swapChainSupport.capabilities.maxImageCount;
            }

            // swap chain config structure
            VkSwapchainCreateInfoKHR createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface = surface;

            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;            // number of layers each image consists of
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            // We'll be drawing on the images in the swap chain from the graphics queue and then submitting them on the presentation queue.
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
            uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            if (indices.graphicsFamily != indices.presentFamily)
            {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;       // avoid changing ownership of images
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            }
            else
            {
                createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;        // most performant
                createInfo.queueFamilyIndexCount = 0; // Optional
                createInfo.pQueueFamilyIndices = nullptr; // Optional
            }

            // no transformation applied
            createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

            // ignore alpha
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;           // we don't care about pixel colors that aren't displayed, enhance performance

            // no old swap chain specified if swap chain needs to be recreated
            createInfo.oldSwapchain = VK_NULL_HANDLE;

            if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create swap chain!");
            }

            // retrieve image handles
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
            swapChainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

            swapChainImageFormat = surfaceFormat.format;
            swapChainExtent = extent;
        }

        void createSurface()
        {
            if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create window surface!");
            }
        }

        void createLogicalDevice()
        {
            // specify queues
            QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            float queuePriority = 1.0f;
            for (uint32_t queueFamily : uniqueQueueFamilies)
            {
                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            VkPhysicalDeviceFeatures deviceFeatures = {};

            // create logical device
            VkDeviceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

            createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
            createInfo.pQueueCreateInfos = queueCreateInfos.data();


            createInfo.pEnabledFeatures = &deviceFeatures;

            // specify extensions and layers
            createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
            createInfo.ppEnabledExtensionNames = deviceExtensions.data();

            if (enableValidationLayers)
            {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            }
            else
            {
                createInfo.enabledLayerCount = 0;
            }

            // instantiate logical device
            if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create logical device!");
            }

            // get graphics queue
            vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
        }

        void pickPhysicalDevice()
        {
	        uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

            if (deviceCount == 0)
            {
                throw std::runtime_error("failed to find GPUs with Vulkan support!");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

            for (const auto& device : devices)
            {
                if (isDeviceSuitable(device))
                {
                    physicalDevice = device;
                    break;
                }
            }

            if (physicalDevice == VK_NULL_HANDLE)
            {
                throw std::runtime_error("failed to find a suitable GPU!");
            }
        }

        bool isDeviceSuitable(VkPhysicalDevice device)
        {
            // we could evaluate this by assigning scores to devices
            // and selecting the highest scoring device

            // the following is not required, but demonstrates how you query devices
            //VkPhysicalDeviceProperties deviceProperties;
            //VkPhysicalDeviceFeatures deviceFeatures;
            //vkGetPhysicalDeviceProperties(device, &deviceProperties);
            //vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
            //return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            //    deviceFeatures.geometryShader;

            QueueFamilyIndices indices = findQueueFamilies(device);

            bool extensionsSupported = checkDeviceExtensionSupport(device);

            bool swapChainAdequate = false;
            if (extensionsSupported)
            {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }

            return indices.isComplete() && extensionsSupported && swapChainAdequate;
        }

        bool checkDeviceExtensionSupport(VkPhysicalDevice device)
        {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

            for (const auto& extension : availableExtensions)
            {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }
  
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
        {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies)
            {
                if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                {
                    indices.graphicsFamily = i;
                }

                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

                if (presentSupport)
                {
                    indices.presentFamily = i;
                }

                if (indices.isComplete())
                {
                   break;
                }

                i++;
            }

	        return indices;
	    }

        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
        {
            SwapChainSupportDetails details;

            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

            if (formatCount != 0)
            {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0)
            {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
        {
            // use SRGB color space if available
            for (const auto& availableFormat : availableFormats)
            {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                {
                    return availableFormat;
                }
            }

            std::cout << "SRGB color format not available" << std::endl;

            // if not just return the first available
            return availableFormats[0];
        }

        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
        {
            // check if we can do triple buffering
            for (const auto& availablePresentMode : availablePresentModes) {
                if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    return availablePresentMode;
                }
            }

            std::cout << "triple buffering (with mailbox presentation mode) not available" << std::endl;

            // guaranteed to be available
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        /**
         * The swap extent is the resolution of the swap chain images and it's almost always exactly equal to the resolution of the window that we're drawing to.
         * @param capabilities
         * @return
         */
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
        {
            if (capabilities.currentExtent.width != UINT32_MAX)
            {
                return capabilities.currentExtent;
            }
            else
            {
                VkExtent2D actualExtent = {WIDTH, HEIGHT};

                actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
                actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

                return actualExtent;
            }
        }

        void setupDebugCallback()
        {
            if (!enableValidationLayers) return;
            VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            createInfo.pfnUserCallback = debugCallback;
            createInfo.pUserData = nullptr;

            if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &callback) != VK_SUCCESS) {
                throw std::runtime_error("failed to set up debug callback!");
            }
        }

        void mainLoop()
        {
            while (!glfwWindowShouldClose(window))
            {
                glfwPollEvents();
            }
        }

        void cleanup()
        {
            for (auto framebuffer : swapChainFramebuffers)
            {
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            }
            
            vkDestroyPipeline(device, graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            vkDestroyRenderPass(device, renderPass, nullptr);
            
            for (auto imageView : swapChainImageViews)
            {
                vkDestroyImageView(device, imageView, nullptr);
            }

            vkDestroySwapchainKHR(device, swapChain, nullptr);
            vkDestroyDevice(device, nullptr);

            if(enableValidationLayers)
            {
                DestroyDebugUtilsMessengerEXT(instance, callback, nullptr);
            }

            vkDestroySurfaceKHR(instance, surface, nullptr);
            vkDestroyInstance(instance, nullptr);
            glfwDestroyWindow(window);
            glfwTerminate();
        }

        void createInstance()
        {
            // check for validation layers if applicable
            if(enableValidationLayers && !checkValidationLayerSupport())
            {
                throw std::runtime_error("validation layers requested, but not available");
            }

            // create vulkan app information structure
            VkApplicationInfo appInfo = {};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = TITLE;
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            auto extensions = getRequiredExtensions();

            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            createInfo.ppEnabledExtensionNames = extensions.data();
            //createInfo.enabledLayerCount = 0;

            if(enableValidationLayers)
            {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            }
            else
            {
                createInfo.enabledLayerCount = 0;
            }

            // actually create the instance and check error
            // vkCreateInstance() returns type VkResult
            if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
                throw std::runtime_error("failed to create instance!");
            }

            checkExtensions();
        }

        void checkExtensions()
        {
            uint32_t extensionCount = 0;
            // query number of extensions by leaving last arg nullptr
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> extensions(extensionCount);
            // query extension details
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
            std::cout << "available extensions: " << std::endl;
            for (const auto& extension : extensions)
            {
                std::cout << "\t" << extension.extensionName << std::endl;
            }
        }

        bool checkValidationLayerSupport()
        {
            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
            //return false;

            for (const char* layerName : validationLayers)
            {
                bool layerFound = false;
                for (const auto& layerProperties : availableLayers)
                {
                    if(strcmp(layerName, layerProperties.layerName) == 0)
                    {
                        layerFound = true;
                        break;
                    }
                }

                if(!layerFound) return false;
            }

            return true;
        }

        std::vector<const char*> getRequiredExtensions()
        {
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

            if(enableValidationLayers)
            {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            return extensions;
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                            const VkDebugUtilsMessengerCallbackDataEXT * pCallbackData, void * pUserData)
        {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }
};

int main()
{
    HelloTriangleApplication app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
