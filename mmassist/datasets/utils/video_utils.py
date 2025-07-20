import os
import av
import math
import io
import base64
import av
from PIL import Image
from typing import Any, Literal

from datasets.arrow_writer import ArrowWriter


def img2str(img: Image.Image):
    with io.BytesIO() as output:
        img.save(output, format="JPEG")
        return base64.b64encode(output.getvalue()).decode("utf-8")


def str2img(img_str: str):
    img_bytes = base64.b64decode(img_str)
    return Image.open(io.BytesIO(img_bytes))


def time_to_frame_index(
    time: float | str, fps: float, rounding: Literal["round", "floor", "ceil"] = "round"
) -> int:
    """Convert a time to a frame index.

    :param time: time in seconds or timestamp in the format "HH:MM:SS.MS"
    :param fps: frames per second
    :param rounding: rounding method

    :return: frame index
    """
    # Parse the timestamp
    if isinstance(time, str):
        timestamp = time
        hours, minutes, seconds = timestamp.split(":")
        seconds, milliseconds = seconds.split(".")

        # Convert to total seconds
        time = (
            (int(hours) * 3600)
            + (int(minutes) * 60)
            + int(seconds)
            + (int(milliseconds) / 1000)
        )

    # Calculate frame index
    if rounding == "round":
        frame_index = round(time * fps)
    elif rounding == "floor":
        frame_index = math.floor(time * fps)
    elif rounding == "ceil":
        frame_index = math.ceil(time * fps)
    else:
        raise ValueError(f"Invalid rounding method: {rounding}")

    return frame_index


def frame_index_to_timestamp(frame_index: int, fps: int, add_ms: bool = False) -> str:
    # Calculate total seconds
    total_seconds = frame_index / fps

    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(total_seconds / 3600)
    minutes = int((total_seconds % 3600) / 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds % 1) * 1000)
    if add_ms:
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:02d}"
    else:
        if milliseconds > 0:
            seconds += 1
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return timestamp


def frame_id_to_filename(
    frame_id: int, prefix: str = "", num_digits: int = 10, extension: str = "jpg"
) -> str:
    return f"{prefix}{frame_id:0{num_digits}d}.{extension}"


def resize_and_crop(img: Image.Image, expected_size: int) -> Image.Image:
    # Calculate the ratio to resize the image to cover the expected size
    ratio = max(expected_size / img.size[0], expected_size / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))

    # Resize the image
    img = img.resize(new_size)

    # Calculate the cropping area
    left = (img.size[0] - expected_size) // 2
    top = (img.size[1] - expected_size) // 2
    right = left + expected_size
    bottom = top + expected_size

    # Crop the image
    img = img.crop((left, top, right, bottom))
    return img


def resize_and_pad(
    img: Image.Image, expected_size: int, fill: Any = (128, 128, 128)
) -> Image.Image:
    # Calculate the new size preserving the aspect ratio
    ratio = min(expected_size / img.size[0], expected_size / img.size[1])
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size)

    # Create a new image with the specified background color and expected size
    new_img = Image.new("RGB", expected_size, fill)

    # Get the position to paste the resized image on the background
    paste_position = (
        (expected_size - new_size[0]) // 2,
        (expected_size - new_size[1]) // 2,
    )
    new_img.paste(img, paste_position)

    return new_img


def extract_frames_to_arrow(
    video_files: list[str],
    output_file: str,
    target_fps: int,
    resize_to_and_crop: int = -1,
    rotate: int = -1,
) -> None:

    # sanity check
    for vf in video_files:
        if not os.path.exists(vf):
            print(f"Video file {vf} does not exist.")
            return None
        try:
            container = av.open(vf)
            container.close()
        except:
            print(f"Failed to open {vf}")
            return None

    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)
    writer = ArrowWriter(path=output_file)
    for vf in video_files:
        # use pyav to extract frames
        container = av.open(vf)
        stream = container.streams.video[0]  # take first video stream
        original_fps = int(stream.average_rate)
        step_size = int((original_fps + 0.5) // target_fps)

        # process the frames and save into an .arrow file
        for ori_idx, frame in enumerate(container.decode(stream)):
            if ori_idx % step_size == 0:
                img = frame.to_image()
                if resize_to_and_crop > 0:
                    img = resize_and_crop(img, resize_to_and_crop)
                if rotate > 0:
                    img = img.rotate(rotate)
                writer.write({"frame": img2str(img)})

        container.close()

    writer.finalize()
