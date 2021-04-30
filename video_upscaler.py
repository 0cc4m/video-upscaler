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
        task_queue, gpu, scale, args, bar, encode_queue,
        upscaler="waifu2x", noise=None):

    while True:
        index = await task_queue.get()

        input_folder = os.path.join(args.directory, f"tmp{index}")
        output_folder = os.path.join(args.directory, f"tmp_out{index}")
        if not os.path.exists(input_folder):
            raise RuntimeError(f"Folder {input_folder} does not exist")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if upscaler == "waifu2x":
            cmd = "waifu2x-ncnn-vulkan"

            cmd_args = ["-i", input_folder, "-o", output_folder, "-g", gpu,
                        "-s", str(scale)]
            if noise:
                cmd_args.extend(["-n", str(noise)])

            proc = await asyncio.create_subprocess_exec(
                cmd,
                *cmd_args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            await proc.wait()

            shutil.rmtree(input_folder)
        elif upscaler == "realsr":
            cmd = "realsr-ncnn-vulkan"

            cmd_args = ["-i", input_folder, "-o", output_folder, "-g", gpu,
                        "-s", str(scale)]

            proc = await asyncio.create_subprocess_exec(
                cmd,
                *cmd_args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            await proc.wait()

            shutil.rmtree(input_folder)
        elif upscaler == "srmd":
            cmd = "srmd-ncnn-vulkan"

            cmd_args = ["-i", input_folder, "-o", output_folder, "-g", gpu,
                        "-s", str(scale)]
            if noise:
                cmd_args.extend(["-n", str(noise)])

            proc = await asyncio.create_subprocess_exec(
                cmd,
                *cmd_args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            await proc.wait()

            shutil.rmtree(input_folder)
        elif upscaler == "vkresample":
            cmd = "VkResample"

            cmd_args = ["-ifolder", input_folder, "-ofolder", output_folder,
                        "-d", gpu, "-numfiles", "500", "-numthreads", "8",
                        "-u", str(scale)]

            proc = await asyncio.create_subprocess_exec(
                cmd,
                *cmd_args,
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

        await encode_queue.put((
            os.path.join(args.directory, f"tmp_out{index}"),
            os.path.join(args.directory, f"tmp{index}.mp4"),
        ))
        task_queue.task_done()
        bar.next()


async def encode_frames(encode_queue, framerate):
    while True:
        input_folder, output_file = await encode_queue.get()

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

        encode_queue.task_done()


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


async def update_bar(bar, batch_index, batch_size, directory):
    bi = batch_index
    i = batch_index * batch_size
    while True:
        batch_dir = os.path.join(directory, f"tmp_out{bi}")
        if os.path.isdir(batch_dir):
            while os.path.isfile(os.path.join(batch_dir, f"{i+1:06}.png")) and\
                    i < batch_size:
                i += 1

            if i >= batch_size:
                bi += 1
                i = 0

            bar.goto(bi * batch_size + i)

        await asyncio.sleep(1)


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
        help="Use RealSR upscaler",
        action="store_true",
    )
    parser.add_argument(
        "--srmd",
        help="Use SRMD upscaler",
        action="store_true",
    )
    parser.add_argument(
        "--vkresample",
        help="Use VkResample upscaler",
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
    parser.add_argument(
        "-n",
        "--noise",
        help="Denoise level",
        default=None,
    )

    args = parser.parse_args()

    input_vid = args.input

    upscaler = "waifu2x"

    if args.realsr:
        upscaler = "realsr"
    elif args.srmd:
        upscaler = "srmd"
    elif args.vkresample:
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

    num_frames = length * framerate

    if not args.resume:
        decode_index = 0
    else:
        decode_index = -1
        # Check for existing video fragments
        for i in range(end_index - 1, -1, -1):
            if os.path.isfile(os.path.join(args.directory, f"tmp{i}.mp4")):
                decode_index = i + 1
                break

    gpus = args.gpu.split(",")

    if not args.scale:
        scale = 4 if args.realsr else 2
    else:
        scale = args.scale

    upscale_queue = asyncio.Queue(maxsize=2 * len(gpus))
    encode_queue = asyncio.Queue()

    bar = EtaBar("Progress", index=decode_index * args.batch_size,
                 max=num_frames)
    bar.update()

    upscale_workers = []

    for gpu in gpus:
        upscale_workers.append(asyncio.create_task(upscale(
            upscale_queue, gpu, scale, args,
            bar, encode_queue, upscaler=upscaler
        )))

    encode_worker = asyncio.create_task(encode_frames(
        encode_queue,
        framerate,
    ))

    bar_updater = asyncio.create_task(update_bar(
        bar,
        decode_index,
        args.batch_size,
        args.directory,
    ))

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    while decode_index < end_index:
        await decode_frames(
            step_length_s * decode_index,
            step_length_s,
            input_vid,
            os.path.join(args.directory, f"tmp{decode_index}"),
        )
        await upscale_queue.put(
            decode_index,
        )
        decode_index += 1

    await upscale_queue.join()

    for worker in upscale_workers:
        worker.cancel()

    await encode_queue.join()

    bar_updater.cancel()
    await bar_updater.join()

    bar.finish()

    encode_worker.cancel()
    await encode_worker.join()

    await asyncio.gather(*upscale_workers, return_exceptions=True)

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
