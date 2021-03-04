#!/usr/bin/env python3
"""
Copyright (C) 2021  0cc4m

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse
import asyncio
from ffmpeg import FFmpeg
import logging
import os
from progress.bar import Bar
import pyprobe
import queue
import shutil

_logger = logging.getLogger(__name__)


class EtaBar(Bar):
    suffix = '%(index)d/%(max)d - ETA: %(eta_td)s'


async def cut_input(start, length, input_file, output_file):
    if not os.path.isfile(input_file):
        raise FileNotFoundError("Input file not found")

    ffmpeg = FFmpeg().option(
        "y",
    ).option(
        "-ss", start
    ).option(
        "-t", length
    ).input(
        input_file,
    ).output(
        output_file,
    )

    @ffmpeg.on("error")
    def on_error(code):
        _logger.error(f"Error: {code}")

    await ffmpeg.execute()


async def decode_frames(start, length, input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    ffmpeg = FFmpeg().option(
        "-ss", start
    ).option(
        "-t", length
    ).input(
        input_file,
    ).output(
        os.path.join(output_folder, "%06d.png"),
    )

    @ffmpeg.on("error")
    def on_error(code):
        _logger.error(f"Error: {code}")

    await ffmpeg.execute()


async def upscale(
        input_folder, output_folder, gpu, scale, upscaler="waifu2x",
        upscaler_args=[]):
    if not os.path.exists(input_folder):
        raise RuntimeError(f"Folder {input_folder} does not exist")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if upscaler == "waifu2x":
        cmd = "waifu2x-ncnn-vulkan"

        args = ["-i", input_folder, "-o", output_folder, "-g", gpu,
                "-s", str(scale)]
        args.extend(upscaler_args)

        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        await proc.wait()

        shutil.rmtree(input_folder)
    elif upscaler == "realsr":
        cmd = "realsr-ncnn-vulkan"

        args = ["-i", input_folder, "-o", output_folder, "-g", gpu,
                "-s", str(scale)]
        args.extend(upscaler_args)

        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        await proc.wait()

        shutil.rmtree(input_folder)
    elif upscaler == "vkresample":
        cmd = "VkResample"

        args = ["-ifolder", input_folder, "-ofolder", output_folder, "-d", gpu,
                "-numfiles", "500", "-numthreads", "8", "-u", str(scale)]
        args.extend(upscaler_args)

        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

        await proc.wait()
    else:
        raise NotImplementedError("Upscaler not implemented")

    if proc.returncode != 0:
        error_lines = []
        while not proc.stderr.at_eof():
            error_lines.append((await proc.stderr.readline()).decode())
        raise RuntimeError("\n".join(error_lines))

    return gpu


async def encode_frames(input_folder, output_file, framerate):
    ffmpeg = FFmpeg().option(
        "y",
    ).option(
        "-r", framerate,
    ).input(
        os.path.join(input_folder, "%06d.png"),
    ).output(
        output_file,
    )

    @ffmpeg.on("error")
    def on_error(code):
        _logger.error(f"Error: {code}")

    await ffmpeg.execute()

    shutil.rmtree(input_folder)


async def combine_video_fragments(output_file, input_fragment_files):
    for f in input_fragment_files:
        if not os.path.isfile(f):
            raise RuntimeError("Input fragment does not exist")

    with open("frags.txt", "w") as f:
        for path in input_fragment_files:
            f.write(f"file '{path}'\n")

    ffmpeg = FFmpeg().option(
        "-f", "concat",
    ).option(
        "-safe", "0",
    ).option('y').input(
        "frags.txt",
    ).output(
        output_file,
    )

    @ffmpeg.on("error")
    def on_error(code):
        _logger.error(f"Error: {code}")

    await ffmpeg.execute()

    os.remove("frags.txt")


async def transplant_audio(input_file, audio_file, output_file):
    if not os.path.exists(input_file):
        raise RuntimeError(f"File {input_file} does not exist")

    if not os.path.exists(audio_file):
        raise RuntimeError(f"File {audio_file} does not exist")

    args = ["-y", "-i", input_file, "-i", audio_file, "-c", "copy", "-map",
            "0:v:0", "-map", "1:a:0", "-shortest", output_file]

    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        *args,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    await proc.wait()

    return proc.returncode


async def main():
    parser = argparse.ArgumentParser(description="VSR Manager")

    parser.add_argument("-i", "--input", help="Input file", required=True)
    parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument(
        "-b", "--batch-size",
        help="Upscaler image batch size",
        type=int,
        default=500,
    )
    parser.add_argument(
        "-d", "--directory", help="Temp file directory", default="vsrmgr_tmp",
    )
    parser.add_argument("-s", "--scale", help="Upscale factor")
    parser.add_argument(
        "--resume",
        help="Attempt to resume unfinished work "
             "(must be same input and parameters)",
        action="store_true",
    )
    parser.add_argument(
        "--realsr",
        help="Use RealSR instead of Waifu2x",
        action="store_true",
    )
    parser.add_argument(
        "--vkresample",
        help="Use VkResample instead of Waifu2x",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--preview",
        help="Create a preview of upscale (format: 'time,length')",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        help="Select specific GPU by id, multiple separated with comma",
        default="0",
    )

    args = parser.parse_args()

    input_vid = args.input

    upscaler = "waifu2x"

    if args.realsr:
        upscaler = "realsr"
    if args.vkresample:
        upscaler = "vkresample"

    if args.preview:
        prev = args.preview.split(",")
        input_vid = args.input.split(".mp4")[0] + "_preview.mp4"
        await cut_input(
            prev[0],
            prev[1],
            args.input,
            input_vid,
        )

    metadata = pyprobe.VideoFileParser().parseFfprobe(input_vid)
    framerate = metadata["videos"][0]["framerate"]
    length = metadata["duration"]
    has_audio = len(metadata["audios"]) > 0

    step_length_s = args.batch_size / framerate

    end_index = int(round(length / step_length_s))

    if not args.resume:
        decode_index = 0
        decode_done_index = 0
        upscale_index = 0
        upscale_done_index = 0
        encode_index = 0
    else:
        decode_index = -1
        decode_done_index = -1
        upscale_index = -1
        upscale_done_index = -1
        encode_index = -1
        # Check for existing video fragments
        for i in range(end_index - 1, -1, -1):
            if os.path.isfile(os.path.join(args.directory, f"tmp{i}.mp4")):
                encode_index = i + 1
                if upscale_index < 0:
                    upscale_index = i + 1
                    upscale_done_index = i + 1
                if decode_index < 0:
                    decode_index = i + 1
                    decode_done_index = i + 1
                break
            if upscale_index < 0 and \
                    os.path.isfile(os.path.join(
                        args.directory, f"tmp_out{i}/{args.batch_size:08}.png"
                    )):
                upscale_index = i + 1
                upscale_done_index = i + 1
            if decode_index < 0 and \
                    os.path.isfile(os.path.join(
                        args.directory, f"tmp{i}/{args.batch_size:08}.png"
                    )):
                decode_index = i + 1
                decode_done_index = i + 1

    gpus = args.gpu.split(",")

    gpu_avail = queue.Queue()
    for gpu in gpus:
        gpu_avail.put(gpu)

    if not args.scale:
        scale = 4 if args.realsr else 2
    else:
        scale = args.scale

    decode_queue = queue.Queue()
    upscale_queue = queue.Queue(maxsize=len(gpus))
    encode_queue = queue.Queue()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    bar = EtaBar("Progress", index=upscale_done_index, max=end_index)
    bar.update()

    while encode_index < end_index:
        while decode_index < end_index and \
                upscale_index >= decode_index - len(gpus):
            decode_queue.put(asyncio.create_task(decode_frames(
                step_length_s * decode_index,
                step_length_s,
                input_vid,
                os.path.join(args.directory, f"tmp{decode_index}"),
            )))
            decode_index += 1

        while upscale_done_index > encode_index:
            encode_queue.put(asyncio.create_task(encode_frames(
                os.path.join(args.directory, f"tmp_out{encode_index}"),
                os.path.join(args.directory, f"tmp{encode_index}.mp4"),
                framerate,
            )))
            encode_index += 1

        while decode_done_index < upscale_done_index + len(gpus) and \
                decode_done_index < decode_index:
            await decode_queue.get()
            decode_done_index += 1

        while upscale_index < decode_done_index and not upscale_queue.full():
            upscale_queue.put(asyncio.create_task(upscale(
                os.path.join(args.directory, f"tmp{upscale_index}"),
                os.path.join(args.directory, f"tmp_out{upscale_index}"),
                gpu_avail.get(),
                scale,
                upscaler=upscaler,
                upscaler_args=[],
            )))
            upscale_index += 1

        if not upscale_queue.empty():
            gpu_avail.put(await upscale_queue.get())
            upscale_done_index += 1
            bar.next()

    while not encode_queue.empty():
        await encode_queue.get()

    bar.finish()

    if end_index > 1:
        await combine_video_fragments(
            os.path.join(args.directory, "tmp_combined.mp4"),
            [os.path.join(args.directory, f"tmp{i}.mp4")
             for i in range(0, end_index)]
        )
    else:
        os.rename(
            os.path.join(args.directory, "tmp0.mp4"),
            os.path.join(args.directory, "tmp_combined.mp4"),
        )

    if has_audio:
        await transplant_audio(
            os.path.join(args.directory, "tmp_combined.mp4"),
            input_vid,
            args.output,
        )
    else:
        os.rename(
            os.path.join(args.directory, "tmp_combined.mp4"),
            args.output,
        )

    # Make sure output exists before cleaning up workspace
    if not os.path.isfile(args.output):
        raise RuntimeError("Output file was not created")

    shutil.rmtree(args.directory)

    if args.preview:
        os.remove(input_vid)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
