# video-upscaler

**video-upscaler** is a Python asyncio-based script which automates extracting images from a videofile, upscaling them with waifu2x-ncnn-vulkan, realsr-ncnn-vulkan, srmd-ncnn-vulkan or VkResample, putting them back together and adding the original audio to the result.

## Requirements
### Python packages
* progress
* PyProbe
* python-ffmpeg

### Other
* ffmpeg
* [realsr-ncnn-vulkan](https://github.com/nihui/realsr-ncnn-vulkan)
* [waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan)
* [srmd-ncnn-vulkan](https://github.com/nihui/srmd-ncnn-vulkan)
* [VkResample](https://github.com/DTolm/VkResample)
