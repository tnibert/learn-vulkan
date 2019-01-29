#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
//#include <vulkan/vulkan.h>

#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>

const int WIDTH = 800;
const int HEIGHT = 600;
const char * TITLE = "Hello Triangle";
const std::vector<const char*> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

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

            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            createInfo.enabledExtensionCount = glfwExtensionCount;
            createInfo.ppEnabledExtensionNames = glfwExtensions;
            createInfo.enabledLayerCount = 0;

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
