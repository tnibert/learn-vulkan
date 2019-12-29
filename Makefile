CFLAGS = -std=c++17 -I/usr/include/vulkan/

LDFLAGS = -L/usr/lib/x86_64-linux-gnu/ `pkg-config --static --libs glfw3` -lvulkan

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)
	glslc shaders/shader.vert -o shaders/vert.spv
	glslc shaders/shader.frag -o shaders/frag.spv

.PHONY: test clean

test: VulkanTest
	VK_LAYER_PATH=/usr/share/vulkan/explicit_layer.d ./VulkanTest

clean:
	rm -f VulkanTest
	rm -f shaders/vert.spv
	rm -f shaders/frag.spv
